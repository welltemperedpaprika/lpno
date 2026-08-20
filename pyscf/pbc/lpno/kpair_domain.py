#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Yu Hsuan Liang
#         Gengzhi Yang
#         Hong-Zhou Ye
#         Timothy C. Berkelbach
#

import numpy as np
from functools import reduce
from scipy.special import erfc

from pyscf import lib
from pyscf.lib import logger

from pyscf.lpno.pno import get_pno1
from pyscf.lpno.tools import zdotCNtoR, safe_eigh

einsum = lib.einsum


def map_ref_R_to_global_lo(
    ref_lo_idx, R_cell_serial_idx, nlo_per_cell, num_cells_in_supercell
):
    """global_idx = R_cell * nlo_per_cell + ref_lo_idx"""
    return R_cell_serial_idx * nlo_per_cell + ref_lo_idx


def map_global_lo_to_ref_R(global_lo_idx, nlo_per_cell, nlo):
    """global_idx -> (ref_lo_idx, R_cell_idx)"""
    return global_lo_idx % nlo_per_cell, global_lo_idx // nlo_per_cell


def sum_periodic_pair_energy(data, nlo_per_cell):
    """Sum reference-cell pair energies with weight 1.0 on diagonal and 0.5 off-diagonal."""
    if data is None:
        return 0.0
    if isinstance(data, dict):
        return sum(float(np.real(val)) * (1.0 if i == j else 0.5)
                   for (i, j), val in data.items())
    arr = np.asarray(data)
    if arr.ndim == 2:
        return float(np.real((arr.sum() + np.trace(arr[:, :nlo_per_cell])) * 0.5))
    return float(np.real(np.sum(arr)))


class PeriodicPairIndex:
    """Index algebra for reference-cell periodic pair storage.

    The supercell code serializes k-mesh translation cells as integers, but the
    arithmetic is multi-dimensional.  ``cell_sub[a, b]`` is the PySCF/PBC
    conservation map for cell ``a - b`` in that serialized ordering.
    """

    def __init__(self, nlo_per_cell, nlo, cell_sub):
        self.nlo_per_cell = int(nlo_per_cell)
        self.nlo = int(nlo)
        if self.nlo_per_cell <= 0 or self.nlo % self.nlo_per_cell != 0:
            raise ValueError("nlo must be a positive multiple of nlo_per_cell")
        self.ncell = self.nlo // self.nlo_per_cell
        self.cell_sub = np.asarray(cell_sub, dtype=int)
        if self.cell_sub.shape != (self.ncell, self.ncell):
            raise ValueError(
                "cell_sub shape must be (ncell, ncell), got "
                f"{self.cell_sub.shape} for ncell={self.ncell}"
            )

    def ref_cell(self, global_lo_idx):
        return map_global_lo_to_ref_R(
            global_lo_idx, self.nlo_per_cell, self.nlo)

    def validate_ref_pair_mask(self, pair_mask, label="pair",
                               require_diagonal=False):
        pair_mask = np.asarray(pair_mask)
        expected = (self.nlo_per_cell, self.nlo)
        if pair_mask.shape != expected:
            raise ValueError(
                f"{label} mask shape must be {expected}, got {pair_mask.shape}"
            )
        if pair_mask.dtype != np.bool_:
            raise ValueError(f"{label} mask must have boolean dtype")
        if require_diagonal:
            diag = np.arange(self.nlo_per_cell)
            if not np.all(pair_mask[diag, diag]):
                raise ValueError(
                    f"{label} mask must include reference-cell diagonal pairs"
                )
        return pair_mask

    def relative_cell(self, target_cell, origin_cell):
        return self.cell_sub[target_cell, origin_cell]

    def inverse_cell(self, cell_idx):
        return int(self.relative_cell(0, cell_idx))

    def relative_lo(self, global_lo_idx, origin_cell):
        lo_ref, cell_idx = self.ref_cell(global_lo_idx)
        rel_cell = self.relative_cell(cell_idx, origin_cell)
        return rel_cell * self.nlo_per_cell + lo_ref

    def shifted_ref_pair(self, i_ref, j_global, target_cell):
        j_ref, j_cell = self.ref_cell(j_global)
        i_shifted = i_ref + target_cell * self.nlo_per_cell
        j_shifted_cell = self.relative_cell(j_cell, self.inverse_cell(target_cell))
        j_shifted = j_shifted_cell * self.nlo_per_cell + j_ref
        return i_shifted, j_shifted

    def expand_ref_pair_mask(self, pair_mask):
        pair_mask = self.validate_ref_pair_mask(pair_mask)

        pair_mask_full = np.zeros((self.nlo, self.nlo), dtype=bool)
        i_refs, j_globals = np.where(pair_mask)
        for target_cell in range(self.ncell):
            i_shifted, j_shifted = self.shifted_ref_pair(
                i_refs, j_globals, target_cell)
            pair_mask_full[np.maximum(i_shifted, j_shifted),
                           np.minimum(i_shifted, j_shifted)] = True
        np.fill_diagonal(pair_mask_full, True)
        return pair_mask_full


def _get_periodic_pair_index(owner):
    pair_index = getattr(owner, "pair_index", None)
    if pair_index is None:
        pair_index = PeriodicPairIndex(
            owner.nlo_per_cell, owner.nlo, owner.cell_sub)
        owner.pair_index = pair_index
    return pair_index


def _minimum_image_vectors(Rvecpair, lattice_vectors):
    """Fold Cartesian pair vectors into the minimum-image cell."""
    lat = lattice_vectors.T
    inv_lat = np.linalg.inv(lat)
    frac_coords = np.dot(Rvecpair, inv_lat.T)
    frac_coords -= np.rint(frac_coords)
    return np.dot(frac_coords, lat.T)


def get_kdosv_eris(osv, eris, near_list, with_K=True, with_J=True, max_memory=None):
    """Calculate dOSV-type ERIs for cell 0 and other cells, assume everything interacts
        K[i,j,R_rel](a_i, b_j) = (i a_i | j b_j)
        J[i,j,R_rel](a_i, b_j) = (j a_i | i b_i)

    Args:
        with_K/J:
            If set to False, None will be returned for K/J

    Returns:
        K, J
    """
    log = logger.new_logger(eris)
    if max_memory is None:
        max_memory = eris.max_memory

    dsize = 8

    nocc = osv.nocc
    nosv = osv.nosv
    if not (with_J or with_K):
        return None, None

    pair_shape = [
        (nosv[i], nosv[j]) if i in near_list and j in near_list else (0, 0)
        for i in range(nocc)
        for j in range(i + 1)
    ]
    mem_est = sum([np.prod(x) for x in pair_shape]) * dsize / 1e6 * 2

    mem_avail = max_memory - lib.current_memory()[0]
    mem_blk = 2 * eris.nvir * eris.naux * dsize / 1e6
    occ_blksize = max(
        1, min(nocc, int(np.floor((mem_avail - mem_est) * 0.7 / mem_blk)))
    )
    log.debug1("occ_blksize for dOSV ERI: %d/%d", occ_blksize, nocc)
    J = {} if with_J else None
    K = {} if with_K else None

    _use_cache = getattr(eris, "_osv_cache_R", None) is not None
    _use_batched_dosv = _use_cache and hasattr(eris, "_project_cached_osv_blk_batch")

    if _use_batched_dosv:
        unique_keys = set()
        for i, j in near_list:
            unique_keys.add((i, i))
            if i != j:
                if with_K:
                    unique_keys.add((j, j))
                if with_J:
                    unique_keys.add((j, i))
                    unique_keys.add((i, j))

        key_list = list(unique_keys)
        occ_list = [k[0] for k in key_list]
        orb_list = [k[1] for k in key_list]
        _proj_cache = eris._project_cached_osv_blk_batch(occ_list, orb_list)

    for i, j in near_list:
        if _use_batched_dosv:
            iALR, iALI = _proj_cache[(i, i)]
        elif _use_cache:
            iALR, iALI = eris._project_cached_osv_blk(i, i)
        else:
            ivLR, ivLI = eris.get_occ_blk(i, i + 1)
            ivLR, ivLI = ivLR[0], ivLI[0]
            iALR = lib.dot(osv.u[i].T.conj(), ivLR)
            iALI = lib.dot(osv.u[i].T.conj(), ivLI)

        if j == i:
            Kii = np.empty((nosv[i], nosv[i]))
            zdotCNtoR(iALR, iALI, iALR.T, iALI.T, cR=Kii)
            if with_K:
                K[i, j] = Kii
            if with_J:
                J[i, j] = Kii
        else:
            if _use_batched_dosv:
                if with_K:
                    jBLR, jBLI = _proj_cache[(j, j)]
                    K[i, j] = np.empty((nosv[i], nosv[j]))
                    zdotCNtoR(iALR, iALI, jBLR.T, jBLI.T, cR=K[i, j])
                if with_J:
                    jALR, jALI = _proj_cache[(j, i)]
                    iBLR, iBLI = _proj_cache[(i, j)]
                    J[i, j] = np.empty((nosv[i], nosv[j]))
                    zdotCNtoR(jALR, jALI, iBLR.T, iBLI.T, cR=J[i, j])
            elif _use_cache:
                if with_K:
                    jBLR, jBLI = eris._project_cached_osv_blk(j, j)
                    K[i, j] = np.empty((nosv[i], nosv[j]))
                    zdotCNtoR(iALR, iALI, jBLR.T, jBLI.T, cR=K[i, j])
                if with_J:
                    jALR, jALI = eris._project_cached_osv_blk(j, i)
                    iBLR, iBLI = eris._project_cached_osv_blk(i, j)
                    J[i, j] = np.empty((nosv[i], nosv[j]))
                    zdotCNtoR(jALR, jALI, iBLR.T, iBLI.T, cR=J[i, j])
            else:
                jvLR, jvLI = eris.get_occ_blk(j, j + 1)
                jvLR, jvLI = jvLR[0], jvLI[0]
                if with_K:
                    jBLR = lib.dot(osv.u[j].T.conj(), jvLR)
                    jBLI = lib.dot(osv.u[j].T.conj(), jvLI)
                    K[i, j] = np.empty((nosv[i], nosv[j]))
                    zdotCNtoR(iALR, iALI, jBLR.T, jBLI.T, cR=K[i, j])
                if with_J:
                    jALR = lib.dot(osv.u[i].T.conj(), jvLR)
                    jALI = lib.dot(osv.u[i].T.conj(), jvLI)
                    iBLR = lib.dot(osv.u[j].T.conj(), ivLR)
                    iBLI = lib.dot(osv.u[j].T.conj(), ivLI)
                    J[i, j] = np.empty((nosv[i], nosv[j]))
                    zdotCNtoR(jALR, jALI, iBLR.T, iBLI.T, cR=J[i, j])
    return K, J



def get_kpsv_eri(
    osv,
    eris,
    pair_mask,
    occ_energy,
    vir_energy,
    pno_param=None,
    nlo_per_cell=None,
    nlo=None,
    thresh_psv_lindep=1e-6,
    thresh_weakpair=0,
    with_ex_ene=True,
    compress_diagpair=False,
    keep_weak_domains=False,
):
    """
    Calculate PSV/PNO ERIs for k-point calculations by looping over reference
    cell orbitals `i` and all supercell orbitals `j`.
    """
    with_pno = pno_param is not None
    nosv = osv.nosv

    strong_pair_mask = np.zeros((nlo_per_cell, nlo), dtype=bool)
    es = [{} for _ in range(nlo_per_cell)]
    ws = [{} for _ in range(nlo_per_cell)]
    Ks = [{} for _ in range(nlo_per_cell)]
    epair_psv_ss = np.zeros((nlo_per_cell, nlo))
    epair_psv_os = np.zeros((nlo_per_cell, nlo))
    epair_pno_ss = np.zeros((nlo_per_cell, nlo))
    epair_pno_os = np.zeros((nlo_per_cell, nlo))

    _use_cache = getattr(eris, "_osv_cache_R", None) is not None

    _use_batched = _use_cache and hasattr(eris, "_project_osv_blk_batch")
    batched_records = []

    def finish_pair(i, j, fac, epsv, wpsv, Kpsv):
        t2psv = Kpsv / (occ_energy[i] + occ_energy[j] - (epsv[:, None] + epsv))
        ed = einsum("ab,ab->", t2psv, Kpsv).real
        ex = -einsum("ab,ba->", t2psv, Kpsv).real if with_ex_ene else 0
        eij_psv_os = ed * fac
        eij_psv_ss = (ed + ex) * fac
        eij_psv = eij_psv_os + eij_psv_ss

        if abs(eij_psv) < thresh_weakpair:
            epair_psv_os[i, j] = eij_psv_os
            epair_psv_ss[i, j] = eij_psv_ss
            strong_pair_mask[i, j] = False
            if keep_weak_domains:
                es[i][j], ws[i][j], Ks[i][j] = epsv, wpsv, Kpsv
            return

        strong_pair_mask[i, j] = True
        if with_pno:
            epno, upno = get_pno1(Kpsv, t2psv, epsv, **pno_param)

            wpno = lib.dot(wpsv, upno)
            Kpno = reduce(lib.dot, (upno.T.conj(), Kpsv, upno))

            Tpno = Kpno / (
                occ_energy[i] + occ_energy[j] - (epno[:, None] + epno)
            )
            ed_pno = einsum("ab,ab->", Tpno, Kpno).real
            ex_pno = -einsum("ab,ba->", Tpno, Kpno).real if with_ex_ene else 0
            eij_pno_os = ed_pno * fac
            eij_pno_ss = (ed_pno + ex_pno) * fac

            epair_pno_os[i, j] = eij_psv_os - eij_pno_os
            epair_pno_ss[i, j] = eij_psv_ss - eij_pno_ss

            es[i][j], ws[i][j], Ks[i][j] = epno, wpno, Kpno
        else:
            es[i][j], ws[i][j], Ks[i][j] = epsv, wpsv, Kpsv

    for i in range(nlo_per_cell):
        if not _use_cache:
            ivLR, ivLI = eris.get_occ_blk(i, i + 1)
            ivLR, ivLI = ivLR[0], ivLI[0]

        pair_data = []
        for j in range(nlo):
            if not pair_mask[i, j]:
                continue
            fac = 1 if i == j else 2

            if i == j and not compress_diagpair:
                es[i][j] = osv.e[i]
                ws[i][j] = np.eye(osv.u[i].shape[1])
                if _use_cache:
                    iALR, iALI = eris._project_cached_osv_blk(i, i)
                else:
                    iALR = lib.dot(osv.u[i].T.conj(), ivLR)
                    iALI = lib.dot(osv.u[i].T.conj(), ivLI)
                Kii = np.empty((nosv[i], nosv[i]))
                zdotCNtoR(iALR, iALI, iALR.T, iALI.T, cR=Kii)
                Ks[i][j] = Kii
                strong_pair_mask[i, j] = True
                continue

            s = np.eye(nosv[i] + nosv[j])
            s_ij = osv.S[max(i, j), min(i, j)]
            if i > j:
                s[: nosv[i], nosv[i] :] = s_ij
                s[nosv[i] :, : nosv[i]] = s_ij.T.conj()
            else:
                s[: nosv[i], nosv[i] :] = s_ij.T.conj()
                s[nosv[i] :, : nosv[i]] = s_ij
            f = np.diag(np.hstack((osv.e[i], osv.e[j])))
            f_ij = osv.F[max(i, j), min(i, j)]
            if i > j:
                f[: nosv[i], nosv[i] :] = f_ij
                f[nosv[i] :, : nosv[i]] = f_ij.T.conj()
            else:
                f[: nosv[i], nosv[i] :] = f_ij.T.conj()
                f[nosv[i] :, : nosv[i]] = f_ij

            epsv, wpsv = safe_eigh(f, s, lindep_thr=thresh_psv_lindep)
            ni = nosv[i]
            components = [(i, wpsv[:ni]), (j, wpsv[ni:])]
            pair_data.append((j, fac, epsv, wpsv, ni, components))

        if _use_batched:
            for j, fac, epsv, wpsv, ni, components in pair_data:
                batched_records.append({
                    "i": i,
                    "j": j,
                    "fac": fac,
                    "epsv": epsv,
                    "wpsv": wpsv,
                    "ni": ni,
                    "components": components,
                    "i_block": None,
                    "j_block": None,
                })
            continue

        for j, fac, epsv, wpsv, ni, components in pair_data:
            if _use_cache:
                iALR, iALI = eris._project_osv_blk(i, components)
                jBLR, jBLI = eris._project_osv_blk(j, components)
            else:
                upsv = lib.dot(osv.u[i], wpsv[:ni]) + lib.dot(osv.u[j], wpsv[ni:])
                iALR = lib.dot(upsv.T.conj(), ivLR)
                iALI = lib.dot(upsv.T.conj(), ivLI)
                jvLR, jvLI = eris.get_occ_blk(j, j + 1)
                jvLR, jvLI = jvLR[0], jvLI[0]
                jBLR = lib.dot(upsv.T.conj(), jvLR)
                jBLI = lib.dot(upsv.T.conj(), jvLI)
            Kpsv = np.empty((iALR.shape[0], jBLR.shape[0]))
            zdotCNtoR(iALR, iALI, jBLR.T, jBLI.T, cR=Kpsv)
            finish_pair(i, j, fac, epsv, wpsv, Kpsv)

    if _use_batched and batched_records:
        i_groups = {}
        for record in batched_records:
            i_groups.setdefault(record["i"], []).append(record)
        for i in sorted(i_groups):
            records = i_groups[i]
            blocks = eris._project_osv_blk_batch(
                i, [record["components"] for record in records])
            for record, block in zip(records, blocks):
                record["i_block"] = block

        j_groups = {}
        for record in batched_records:
            j_groups.setdefault(record["j"], []).append(record)
        for j in sorted(j_groups):
            records = j_groups[j]
            blocks = eris._project_osv_blk_batch(
                j, [record["components"] for record in records])
            for record, block in zip(records, blocks):
                record["j_block"] = block

        for record in batched_records:
            i = record["i"]
            j = record["j"]
            iALR, iALI = record["i_block"]
            jBLR, jBLI = record["j_block"]
            Kpsv = np.empty((iALR.shape[0], jBLR.shape[0]))
            zdotCNtoR(iALR, iALI, jBLR.T, jBLI.T, cR=Kpsv)
            finish_pair(
                i,
                j,
                record["fac"],
                record["epsv"],
                record["wpsv"],
                Kpsv,
            )

    epair_psv = lib.tag_array(
        epair_psv_ss + epair_psv_os, e_corr_ss=epair_psv_ss, e_corr_os=epair_psv_os
    )
    epair_pno = lib.tag_array(
        epair_pno_ss + epair_pno_os, e_corr_ss=epair_pno_ss, e_corr_os=epair_pno_os
    )

    return strong_pair_mask, es, ws, Ks, epair_psv, epair_pno


def pbc_pair_dipole(
    cell, lo_coeff, nlo_per_cell, moe_lo, vir_coeff, osv, minimum_image=True
):
    nocc = lo_coeff.shape[1]
    dipao = cell.pbc_intor("int1e_r", comp=3)
    lo_r = einsum("xpq,pi,qi->ix", dipao, lo_coeff.conj(), lo_coeff)
    Rvecpair = lo_r[:nlo_per_cell, None, :] - lo_r
    if minimum_image:
        Rvecpair = _minimum_image_vectors(Rvecpair, cell.a)
    Rpair = np.linalg.norm(Rvecpair, axis=-1)
    diplv = einsum("xpq,pi,qa->ixa", dipao, lo_coeff.conj(), vir_coeff)
    diplosv = [einsum("xa,aA->xA", diplv[i], osv.u[i]) for i in range(nlo_per_cell)]
    epair = np.zeros_like(Rpair)
    for i in range(nlo_per_cell):
        for j in range(nocc):
            if Rpair[i, j] < 1e-9:
                continue
            Rbar = Rvecpair[i, j] / Rpair[i, j]
            j_ref, _ = map_global_lo_to_ref_R(j, nlo_per_cell, nocc)
            vab = einsum("xA,xB->AB", diplosv[i], diplosv[j_ref])
            vab -= (
                einsum(
                    "A,B->AB", np.dot(Rbar, diplosv[i]), np.dot(Rbar, diplosv[j_ref])
                )
                * 3
            )
            tab = vab.conj() / (
                moe_lo[i] + moe_lo[j] - (osv.e[i][:, None] + osv.e[j_ref])
            )
            eij = einsum("ab,ab->", tab, vab) / Rpair[i, j] ** 6 * 4
            epair[i, j] = eij
    return epair, Rpair


def _ewald_dipole_tensor(R, eta, Ls, Gv, vol):
    """Build the Ewald-summed dipole-dipole interaction tensor T_αβ(R).

    T(R) = T_real(R) + T_recip(R), where:
      T_real_αβ = Σ_L [ A(|R+L|) δ_αβ - B(|R+L|) (R+L)_α(R+L)_β / |R+L|² ]
      T_recip_αβ = (4π/Ω) Σ_{G≠0} (G_α G_β / G²) exp(-G²/4η²) cos(G·R)

    with A(r) = erfc(ηr)/r³ + 2η/√π exp(-η²r²)/r²
         B(r) = 3 erfc(ηr)/r³ + 2η/√π (3+2η²r²) exp(-η²r²)/r²
    """
    T = np.zeros((3, 3))
    eta2 = eta * eta
    two_eta_sqrtpi = 2.0 * eta / np.sqrt(np.pi)

    # Real-space sum
    for L in Ls:
        d = R + L
        r2 = np.dot(d, d)
        if r2 < 1e-20:
            continue
        r = np.sqrt(r2)
        r3 = r * r2
        exp_term = np.exp(-eta2 * r2)
        erfc_term = erfc(eta * r)

        A_r = erfc_term / r3 + two_eta_sqrtpi * exp_term / r2
        B_r = (
            3.0 * erfc_term / r3
            + two_eta_sqrtpi * (3.0 + 2.0 * eta2 * r2) * exp_term / r2
        )
        rhat = np.outer(d, d) / r2
        T += A_r * np.eye(3) - B_r * rhat

    # Reciprocal-space sum (G≠0)
    G2 = np.sum(Gv * Gv, axis=1)
    mask = G2 > 1e-20
    Gv_nz = Gv[mask]
    G2_nz = G2[mask]
    GdotR = Gv_nz @ R
    prefac = (4.0 * np.pi / vol) * np.exp(-G2_nz / (4.0 * eta2)) * np.cos(GdotR) / G2_nz
    # T_recip = Σ prefac[g] * G[g] ⊗ G[g]
    T += np.einsum("g,gx,gy->xy", prefac, Gv_nz, Gv_nz)

    return T


def pbc_pair_ewald_dipole(cell, lo_coeff, nlo_per_cell, moe_lo, vir_coeff, osv):
    """Ewald-summed dipole approximation for periodic pair energies.

    Uses the Ewald-decomposed dipole-dipole interaction tensor instead of
    the molecular 1/R³ formula. See Appendix A of draft, Eqs. A7-A11.

    The Ewald lattice sums are performed over the supercell lattice, which
    matches the periodicity of the k-space GDF Coulomb kernel.

    Returns:
        epair: (nlo_per_cell, nocc) pair energies
        Rpair: (nlo_per_cell, nocc) minimum-image distances
    """
    nocc = lo_coeff.shape[1]

    # Ewald parameters (supercell)
    eta, _ = cell.get_ewald_params()
    chargs = cell.atom_charges()
    log_precision = np.log(cell.precision / (chargs.sum() ** 2 * 16 * np.pi**2))
    ke_cutoff = -2 * eta**2 * log_precision
    mesh = cell.cutoff_to_mesh(ke_cutoff)
    Gv = cell.get_Gv(mesh)
    Ls = cell.get_lattice_Ls()
    vol = cell.vol

    # LO dipole integrals
    dipao = cell.pbc_intor("int1e_r", comp=3)
    lo_r = einsum("xpq,pi,qi->ix", dipao, lo_coeff.conj(), lo_coeff)
    Rvecpair = lo_r[:nlo_per_cell, None, :] - lo_r
    Rvecpair = _minimum_image_vectors(Rvecpair, cell.a)
    Rpair = np.linalg.norm(Rvecpair, axis=-1)

    # LO-OSV dipole moments
    diplv = einsum("xpq,pi,qa->ixa", dipao, lo_coeff.conj(), vir_coeff)
    diplosv = [einsum("xa,aA->xA", diplv[i], osv.u[i]) for i in range(nlo_per_cell)]

    epair = np.zeros_like(Rpair)
    for i in range(nlo_per_cell):
        for j in range(nocc):
            if Rpair[i, j] < 1e-9:
                continue
            R = Rvecpair[i, j]
            T = _ewald_dipole_tensor(R, eta, Ls, Gv, vol)
            j_ref, _ = map_global_lo_to_ref_R(j, nlo_per_cell, nocc)
            # vab_AB = Σ_αβ D_i^αA T_αβ D_j^βB
            vab = einsum("xA,xy,yB->AB", diplosv[i], T, diplosv[j_ref])
            tab = vab.conj() / (
                moe_lo[i] + moe_lo[j] - (osv.e[i][:, None] + osv.e[j_ref])
            )
            eij = einsum("ab,ab->", tab, vab) * 4
            epair[i, j] = eij
    return epair, Rpair


class PairDomain_kdOSV(lib.StreamObject):
    """
    Attributes:
        eris:
        osv :
        pair_mask (ndarray): A boolean array of shape (nlo_per_cell, nlo) indicating
                             which pairs are treated as strong pairs.
        near_list (list): A list of (i, j) tuples for strong pairs.
        nlo (int): Total number of localized orbitals in the supercell.
        nlo_per_cell (int): Number of localized orbitals per unit cell.
        npair (int): The number of strong pairs.
        K (dict): (i a_i | j b_j)
        J (dict): (j a_i | i b_j)
    """

    def __init__(
        self,
        eris,
        osv,
        pair_mask,
        nlo_per_cell,
        nlo,
        near_list,
        with_J=True,
        s1e=None,
        vir_coeff=None,
        cell_sub=None,
    ):
        self.eris = eris
        self.osv = osv
        self.pair_mask = pair_mask
        self.nlo = nlo
        self.nlo_per_cell = nlo_per_cell
        self.near_list = near_list
        self.npair = len(self.near_list)
        self.cell_sub = cell_sub
        self.pair_index = PeriodicPairIndex(nlo_per_cell, nlo, cell_sub)
        self.s1e = s1e
        self.vir_coeff = vir_coeff
        self.K = None
        self.J = None
        self.build(with_J)

    @property
    def S(self):
        """S[i,j](a_i, b_j) = <a_i|b_j>."""
        return self.osv.get_ovlp()

    def build(self, with_J):
        self.K, self.J = get_kdosv_eris(
            self.osv, self.eris, self.near_list, with_J=with_J
        )

    def loop_pair(self):
        return self.near_list

    def new_array(self):
        return {}

    def loop_k(self, i, j):
        """k indices contributing to R_ij = K_ij - sum_k (f_kj*t[i,k]*S + f_ik*S*t[k,j])"""
        pair_index = _get_periodic_pair_index(self)
        j_ref, R_idx_j = pair_index.ref_cell(j)
        all_k = np.arange(self.nlo)
        all_k_ref = all_k % self.nlo_per_cell
        all_R_idx_k = all_k // self.nlo_per_cell

        all_R_idx_j_shifted = pair_index.relative_cell(
            R_idx_j, all_R_idx_k)
        all_j_shifted = all_R_idx_j_shifted * self.nlo_per_cell + j_ref

        mask_i = self.pair_mask[i, :]
        mask_j = self.pair_mask[all_k_ref, all_j_shifted]
        return np.where(mask_i | mask_j)[0]


class PairDomain_kPNO(lib.StreamObject):
    """
    PNO-based pair domain for k-point calculations.

    Attributes:
        osv (OSV object):
            Provides osv.u, osv.e, osv.nosv.
        npsv (1D list of length nocc):
            Number of PSVs for each occupied.
        w :
            Joint OSV-to-PSV transformation matrix.
        e :
            PSV energy (pseudocanonicalized).
        K :
            K[i,j](a_ij, b_ij) := (i a_ij | j b_ij)
    """

    def __init__(
        self,
        eris,
        osv,
        pno_param,
        occ_energy,
        vir_energy,
        pair_mask=None,
        thresh_psv_lindep=1e-6,
        thresh_weakpair=0,
        with_ex_ene=True,
        compress_diagpair=False,
        nlo_per_cell=None,
        nlo=None,
        cell_sub=None,
        keep_weak_domains=False,
    ):
        self.osv = osv
        self.eris = eris
        self.pno_param = pno_param
        self.occ_energy = occ_energy
        self.vir_energy = vir_energy
        self.pair_mask = pair_mask
        self.thresh_psv_lindep = thresh_psv_lindep
        self.thresh_weakpair = thresh_weakpair
        self.with_ex_ene = with_ex_ene
        self.compress_diagpair = compress_diagpair
        self.nlo_per_cell = nlo_per_cell
        self.nlo = nlo
        self.cell_sub = cell_sub
        self.pair_index = PeriodicPairIndex(nlo_per_cell, nlo, cell_sub)
        self.keep_weak_domains = keep_weak_domains

        self.e = None
        self.w = None
        self.K = None
        self.pair_mask_strong = None
        self.epair_pno = None
        self.epair_psv = None

        self._ovlp_cache = {}
        self._ovlp_cache_enabled = True

        self.build()

    def build(self):
        builder = get_kpsv_eri
        (
            self.pair_mask_strong,
            self.e,
            self.w,
            self.K,
            self.epair_psv,
            self.epair_pno,
        ) = builder(
            self.osv,
            self.eris,
            self.pair_mask,
            self.occ_energy,
            self.vir_energy,
            pno_param=self.pno_param,
            nlo_per_cell=self.nlo_per_cell,
            nlo=self.nlo,
            thresh_psv_lindep=self.thresh_psv_lindep,
            thresh_weakpair=self.thresh_weakpair,
            with_ex_ene=self.with_ex_ene,
            compress_diagpair=self.compress_diagpair,
            keep_weak_domains=self.keep_weak_domains,
        )

        self.epair_psv_ss = self.epair_psv.e_corr_ss
        self.epair_psv_os = self.epair_psv.e_corr_os

        if self.epair_pno is not None:
            self.epair_pno_ss = self.epair_pno.e_corr_ss
            self.epair_pno_os = self.epair_pno.e_corr_os

        self.npsv = np.zeros((self.nlo_per_cell, self.nlo), dtype=np.int32)
        self.pair_shape = {}
        for i in range(self.nlo_per_cell):
            for j in range(self.nlo):
                if self.pair_mask_strong[i, j]:
                    n_pno = self.e[i][j].size
                    self.npsv[i, j] = n_pno
                    self.pair_shape[(i, j)] = (n_pno, n_pno)
                else:
                    self.pair_shape[(i, j)] = (0, 0)

    def loop_k(self, i, j, foo_mask=None):
        pair_index = _get_periodic_pair_index(self)
        j_ref, R_idx_j = pair_index.ref_cell(j)
        all_k = np.arange(self.nlo)
        all_k_ref = all_k % self.nlo_per_cell
        all_R_idx_k = all_k // self.nlo_per_cell

        all_R_idx_j_shifted = pair_index.relative_cell(
            R_idx_j, all_R_idx_k)
        all_j_shifted = all_R_idx_j_shifted * self.nlo_per_cell + j_ref

        pair_mask_residual = getattr(self, "pair_mask_residual", self.pair_mask_strong)
        mask_i = pair_mask_residual[i, :]
        mask_j = pair_mask_residual[all_k_ref, all_j_shifted]

        mask = mask_i | mask_j

        if foo_mask is not None:
            mask = mask & (foo_mask[:, j] | foo_mask[i, :])

        return np.where(mask)[0]

    def clear_ovlp_cache(self):
        self._ovlp_cache.clear()

    def precompute_k_lists(self, foo_mask):
        """{(i,j): k_indices} for all strong pairs."""
        k_lists = {}
        for i, j in self.loop_strong_pairs():
            k_lists[(i, j)] = self.loop_k(i, j, foo_mask)
        return k_lists

    def precompute_overlaps(self, foo_mask, t2_keys):
        """Populate _ovlp_cache with all overlaps needed for the residual."""
        pair_index = _get_periodic_pair_index(self)
        for i, j in self.loop_strong_pairs():
            for k in self.loop_k(i, j, foo_mask):
                # Term 1: needs get_ovlp(i, j, i, k)
                if k != j and foo_mask[k, j] and (i, k) in t2_keys:
                    if (i, j, i, k) not in self._ovlp_cache:
                        self.get_ovlp(i, j, i, k)

                # Term 2: needs get_ovlp(i, j, k, j)
                if k != i and foo_mask[i, k]:
                    k_ref, k_cell = pair_index.ref_cell(k)
                    j_relative = pair_index.relative_lo(j, k_cell)

                    if (k_ref, j_relative) in t2_keys:
                        if (i, j, k, j) not in self._ovlp_cache:
                            self.get_ovlp(i, j, k, j)

    def loop_strong_pairs(self):
        """Yield strong pairs (i, j)."""
        for i in range(self.nlo_per_cell):
            for j in range(self.nlo):
                if self.pair_mask_strong[i, j]:
                    yield i, j

    def get_u(self, i, j):
        """PNO coefficients in AO basis for pair (i, j)."""
        j_ref, _ = map_global_lo_to_ref_R(j, self.nlo_per_cell, self.nlo)

        u = self.osv.u
        if i == j:
            if (i in self.w) and (j in self.w[i]):
                w = self.w[i][j]
                return lib.dot(u[i], w)
            else:
                return u[i]
        else:
            w = self.w[i][j]
            ni = self.osv.nosv[i]
            return lib.dot(u[i], w[:ni]) + lib.dot(u[j], w[ni:])

    def get_ovlp(self, i, j, k, l):
        """<PNO_ij | PNO_kl> overlap matrix."""
        if self._ovlp_cache_enabled:
            cache_key = (i, j, k, l)
            if cache_key in self._ovlp_cache:
                return self._ovlp_cache[cache_key]

        pair_index = _get_periodic_pair_index(self)
        k_ref, k_cell_idx = pair_index.ref_cell(k)
        l_relative = pair_index.relative_lo(l, k_cell_idx)

        osv = self.osv
        wij = self.w[i][j]
        wkl = self.w[k_ref][l_relative]

        if i == j and k == l:
            S_ik = osv.S[max(i, k), min(i, k)]
            if i < k:
                S_ik = S_ik.T.conj()
            result = S_ik

        elif i == j:
            # split wkl at nosv[k_ref] (construction dimension)
            nk = osv.nosv[k_ref]
            w_k = wkl[:nk]
            w_l = wkl[nk:]

            S_ik = osv.S[max(i, k), min(i, k)]
            if i < k:
                S_ik = S_ik.T.conj()

            S_il = osv.S[max(i, l), min(i, l)]
            if i < l:
                S_il = S_il.T.conj()
            result = lib.dot(S_ik, w_k) + lib.dot(S_il, w_l)

        elif k == l:
            ni = osv.nosv[i]
            w_i = wij[:ni]
            w_j = wij[ni:]
            S_ik = osv.S[max(i, k), min(i, k)]
            if i < k:
                S_ik = S_ik.T.conj()

            S_jk = osv.S[max(j, k), min(j, k)]
            if j < k:
                S_jk = S_jk.T.conj()
            result = lib.dot(w_i.T.conj(), S_ik) + lib.dot(w_j.T.conj(), S_jk)

        else:
            ni = osv.nosv[i]
            w_i = wij[:ni]
            w_j = wij[ni:]

            nk = osv.nosv[k_ref]
            w_k = wkl[:nk]
            w_l = wkl[nk:]

            # <PNO_ij|PNO_kl> = w_i^H S_ik w_k + w_i^H S_il w_l
            #                 + w_j^H S_jk w_k + w_j^H S_jl w_l
            S_ik = osv.S[max(i, k), min(i, k)]
            if i < k:
                S_ik = S_ik.T.conj()
            if S_ik.size > 0:
                result = reduce(lib.dot, (w_i.T.conj(), S_ik, w_k))
            else:
                result = np.zeros((w_i.shape[1], w_k.shape[1]), dtype=w_i.dtype)

            S_il = osv.S[max(i, l), min(i, l)]
            if i < l:
                S_il = S_il.T.conj()
            if S_il.size > 0:
                result += reduce(lib.dot, (w_i.T.conj(), S_il, w_l))

            S_jk = osv.S[max(j, k), min(j, k)]
            if j < k:
                S_jk = S_jk.T.conj()
            if S_jk.size > 0:
                result += reduce(lib.dot, (w_j.T.conj(), S_jk, w_k))

            S_jl = osv.S[max(j, l), min(j, l)]
            if j < l:
                S_jl = S_jl.T.conj()
            if S_jl.size > 0:
                result += reduce(lib.dot, (w_j.T.conj(), S_jl, w_l))

        if self._ovlp_cache_enabled:
            self._ovlp_cache[cache_key] = result

        return result

