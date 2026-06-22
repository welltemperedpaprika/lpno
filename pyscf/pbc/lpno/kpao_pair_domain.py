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
# Author: Yu Hsuan Liang <yhljason@berkeley.edu>
#         Gengzhi Yang
#         Hong-Zhou Ye
#         Timothy C. Berkelbach
#

"""Pair-domain builders for periodic PAO-basis OSVs."""

from functools import reduce

import numpy as np
from pyscf import lib

from pyscf.pbc.lpno.kpair_domain import (
    PairDomain_kPNO,
    PairDomain_kdOSV,
    PeriodicPairIndex,
)
from pyscf.lpno.pno import get_pno1
from pyscf.lpno.tools import safe_eigh, zdotCNtoR

einsum = lib.einsum


def _get_pao_osv_projector(osv, pao, orb_idx):
    pao2vir = pao.get_orbital_pao2vir_coeff(orb_idx)
    if osv.u[orb_idx].shape[0] != pao2vir.shape[1]:
        raise ValueError(
            "PAO-basis OSV coefficients must have one row per PAO-domain "
            f"function; got {osv.u[orb_idx].shape[0]} OSV rows for "
            f"{pao2vir.shape[1]} PAO-domain rows"
        )
    return lib.dot(pao2vir, osv.u[orb_idx])


def get_pao_osv_components_projector(osv, pao, components):
    """Form a PSV projector from PAO-basis OSV component weights."""
    projector = None
    for orb_idx, coeff in components:
        orb_projector = lib.dot(_get_pao_osv_projector(osv, pao, orb_idx), coeff)
        if projector is None:
            projector = orb_projector
        else:
            projector = projector + orb_projector
    if projector is None:
        raise ValueError("components must contain at least one OSV component")
    return projector


def _get_projected_blk_batch(eris, occ_idx, projectors):
    if not projectors:
        return []
    if hasattr(eris, "get_projected_blk_batch"):
        return eris.get_projected_blk_batch(occ_idx, projectors)
    return [eris.get_projected_blk(occ_idx, projector)
            for projector in projectors]


def _record_psv_projection_batch(eris, side, occ_idx, projectors):
    if hasattr(eris, "record_pao_psv_projection_batch"):
        eris.record_pao_psv_projection_batch(
            side=side,
            occ_idx=occ_idx,
            projector_count=len(projectors),
            total_projector_cols=sum(projector.shape[1] for projector in projectors),
        )


def get_kdosv_eris_pao_projected(
    osv, eris, pao, near_list, with_K=True, with_J=True
):
    """Build kdOSV ERIs for OSVs represented in local PAO coordinates."""
    if not (with_J or with_K):
        return None, None

    nocc = osv.nocc
    nosv = osv.nosv
    J = {} if with_J else None
    K = {} if with_K else None
    projector_cache = {}
    block_cache = {}

    def projector(orb_idx):
        orb_idx = int(orb_idx)
        if orb_idx not in projector_cache:
            projector_cache[orb_idx] = _get_pao_osv_projector(osv, pao, orb_idx)
        return projector_cache[orb_idx]

    needed_blocks = {}

    def add_needed(occ_idx, orb_idx):
        occ_idx = int(occ_idx)
        orb_idx = int(orb_idx)
        orb_list = needed_blocks.setdefault(occ_idx, [])
        if orb_idx not in orb_list:
            orb_list.append(orb_idx)

    for i, j in near_list:
        if not (0 <= i < nocc and 0 <= j < nocc):
            raise ValueError(f"near_list pair {(i, j)} outside nocc={nocc}")
        add_needed(i, i)
        if i != j:
            add_needed(j, j)
            if with_J:
                add_needed(j, i)
                add_needed(i, j)

    for occ_idx, orb_indices in needed_blocks.items():
        blocks = _get_projected_blk_batch(
            eris, occ_idx, [projector(orb_idx) for orb_idx in orb_indices])
        for orb_idx, block in zip(orb_indices, blocks):
            block_cache[(occ_idx, orb_idx)] = block

    def project(occ_idx, orb_idx):
        block_key = (int(occ_idx), int(orb_idx))
        if block_key not in block_cache:
            block_cache[block_key] = eris.get_projected_blk(
                occ_idx, projector(orb_idx))
        return block_cache[block_key]

    for i, j in near_list:
        iALR, iALI = project(i, i)

        if i == j:
            Kii = np.empty((nosv[i], nosv[i]))
            zdotCNtoR(iALR, iALI, iALR.T, iALI.T, cR=Kii)
            if with_K:
                K[i, j] = Kii
            if with_J:
                J[i, j] = Kii
            continue

        jBLR, jBLI = project(j, j)
        if with_K:
            K[i, j] = np.empty((nosv[i], nosv[j]))
            zdotCNtoR(iALR, iALI, jBLR.T, jBLI.T, cR=K[i, j])

        if with_J:
            jALR, jALI = project(j, i)
            iBLR, iBLI = project(i, j)
            J[i, j] = np.empty((nosv[i], nosv[j]))
            zdotCNtoR(jALR, jALI, iBLR.T, iBLI.T, cR=J[i, j])

    return K, J


def get_kpsv_eri_pao_projected(
    osv,
    eris,
    pao,
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
    """Build periodic PSV/PNO data from PAO-basis OSVs."""
    del vir_energy
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
    orbital_projector_cache = {}
    orbital_block_cache = {}
    pair_records = []

    def orbital_projector(orb_idx):
        orb_idx = int(orb_idx)
        if orb_idx not in orbital_projector_cache:
            orbital_projector_cache[orb_idx] = _get_pao_osv_projector(
                osv, pao, orb_idx)
        return orbital_projector_cache[orb_idx]

    def orbital_projected_block(occ_idx, orb_idx):
        block_key = (int(occ_idx), int(orb_idx))
        if block_key not in orbital_block_cache:
            orbital_block_cache[block_key] = eris.get_projected_blk(
                occ_idx, orbital_projector(orb_idx))
        return orbital_block_cache[block_key]

    def components_projector(components):
        projector = None
        for orb_idx, coeff in components:
            orb_projector = lib.dot(orbital_projector(orb_idx), coeff)
            if projector is None:
                projector = orb_projector
            else:
                projector = projector + orb_projector
        if projector is None:
            raise ValueError("components must contain at least one OSV component")
        return projector

    for i in range(nlo_per_cell):
        for j in range(nlo):
            if not pair_mask[i, j]:
                continue
            fac = 1 if i == j else 2

            if i == j and not compress_diagpair:
                es[i][j] = osv.e[i]
                ws[i][j] = np.eye(osv.u[i].shape[1])
                iALR, iALI = orbital_projected_block(i, i)
                Kii = np.empty((nosv[i], nosv[i]))
                zdotCNtoR(iALR, iALI, iALR.T, iALI.T, cR=Kii)
                Ks[i][j] = Kii
                strong_pair_mask[i, j] = True
                continue

            s = np.eye(nosv[i] + nosv[j])
            s_ij = osv.S[max(i, j), min(i, j)]
            if i > j:
                s[:nosv[i], nosv[i]:] = s_ij
                s[nosv[i]:, :nosv[i]] = s_ij.T.conj()
            else:
                s[:nosv[i], nosv[i]:] = s_ij.T.conj()
                s[nosv[i]:, :nosv[i]] = s_ij

            f = np.diag(np.hstack((osv.e[i], osv.e[j])))
            f_ij = osv.F[max(i, j), min(i, j)]
            if i > j:
                f[:nosv[i], nosv[i]:] = f_ij
                f[nosv[i]:, :nosv[i]] = f_ij.T.conj()
            else:
                f[:nosv[i], nosv[i]:] = f_ij.T.conj()
                f[nosv[i]:, :nosv[i]] = f_ij

            epsv, wpsv = safe_eigh(f, s, lindep_thr=thresh_psv_lindep)
            ni = nosv[i]
            upsv = components_projector([(i, wpsv[:ni]), (j, wpsv[ni:])])
            pair_records.append({
                "i": i,
                "j": j,
                "fac": fac,
                "epsv": epsv,
                "wpsv": wpsv,
                "upsv": upsv,
                "i_block": None,
                "j_block": None,
            })

    for i in range(nlo_per_cell):
        records = [record for record in pair_records if record["i"] == i]
        projectors = [record["upsv"] for record in records]
        blocks = _get_projected_blk_batch(eris, i, projectors)
        _record_psv_projection_batch(eris, "i", i, projectors)
        for record, block in zip(records, blocks):
            record["i_block"] = block

    j_groups = {}
    for record in pair_records:
        j_groups.setdefault(record["j"], []).append(record)
    for j in sorted(j_groups):
        records = j_groups[j]
        projectors = [record["upsv"] for record in records]
        blocks = _get_projected_blk_batch(eris, j, projectors)
        _record_psv_projection_batch(eris, "j", j, projectors)
        for record, block in zip(records, blocks):
            record["j_block"] = block

    for record in pair_records:
        i = record["i"]
        j = record["j"]
        fac = record["fac"]
        epsv = record["epsv"]
        wpsv = record["wpsv"]
        iALR, iALI = record["i_block"]
        jBLR, jBLI = record["j_block"]
        Kpsv = np.empty((iALR.shape[0], jBLR.shape[0]))
        zdotCNtoR(iALR, iALI, jBLR.T, jBLI.T, cR=Kpsv)

        t2psv = Kpsv / (occ_energy[i] + occ_energy[j]
                        - (epsv[:, None] + epsv))
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
            continue

        strong_pair_mask[i, j] = True
        if with_pno:
            epno, upno = get_pno1(Kpsv, t2psv, epsv, **pno_param)
            wpno = lib.dot(wpsv, upno)
            Kpno = reduce(lib.dot, (upno.T.conj(), Kpsv, upno))
            Tpno = Kpno / (occ_energy[i] + occ_energy[j]
                           - (epno[:, None] + epno))
            ed_pno = einsum("ab,ab->", Tpno, Kpno).real
            ex_pno = (-einsum("ab,ba->", Tpno, Kpno).real
                      if with_ex_ene else 0)
            eij_pno_os = ed_pno * fac
            eij_pno_ss = (ed_pno + ex_pno) * fac
            epair_pno_os[i, j] = eij_psv_os - eij_pno_os
            epair_pno_ss[i, j] = eij_psv_ss - eij_pno_ss
            es[i][j], ws[i][j], Ks[i][j] = epno, wpno, Kpno
        else:
            es[i][j], ws[i][j], Ks[i][j] = epsv, wpsv, Kpsv

    epair_psv = lib.tag_array(
        epair_psv_ss + epair_psv_os,
        e_corr_ss=epair_psv_ss,
        e_corr_os=epair_psv_os,
    )
    epair_pno = lib.tag_array(
        epair_pno_ss + epair_pno_os,
        e_corr_ss=epair_pno_ss,
        e_corr_os=epair_pno_os,
    )

    return strong_pair_mask, es, ws, Ks, epair_psv, epair_pno


class PairDomain_kdOSV_PAOBasis(PairDomain_kdOSV):
    """kdOSV pair domain for OSVs stored in PAO-domain coordinates."""

    def __init__(
        self,
        eris,
        osv,
        pao,
        pair_mask,
        nlo_per_cell,
        nlo,
        near_list,
        with_J=True,
        cell_sub=None,
    ):
        self.eris = eris
        self.osv = osv
        self.pao = pao
        self.nlo = nlo
        self.nlo_per_cell = nlo_per_cell
        self.cell_sub = cell_sub
        self.pair_index = PeriodicPairIndex(nlo_per_cell, nlo, cell_sub)
        self.pair_mask = self.pair_index.validate_ref_pair_mask(
            pair_mask, label="pair", require_diagonal=True).copy()
        self.near_list = near_list
        self.npair = len(self.near_list)
        self.K = None
        self.J = None
        self.build(with_J)

    def build(self, with_J):
        self.K, self.J = get_kdosv_eris_pao_projected(
            self.osv, self.eris, self.pao, self.near_list, with_J=with_J)

    @property
    def S(self):
        return self.osv.get_ovlp()


class PairDomain_kPNO_PAOBasis(PairDomain_kPNO):
    """KPNO pair domain for OSVs stored in PAO-domain coordinates."""

    def __init__(
        self,
        eris,
        osv,
        pao,
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
        self.pao = pao
        super().__init__(
            eris,
            osv,
            pno_param,
            occ_energy,
            vir_energy,
            pair_mask=pair_mask,
            thresh_psv_lindep=thresh_psv_lindep,
            thresh_weakpair=thresh_weakpair,
            with_ex_ene=with_ex_ene,
            compress_diagpair=compress_diagpair,
            nlo_per_cell=nlo_per_cell,
            nlo=nlo,
            cell_sub=cell_sub,
            keep_weak_domains=keep_weak_domains,
        )

    def build(self):
        (
            self.pair_mask_strong,
            self.e,
            self.w,
            self.K,
            self.epair_psv,
            self.epair_pno,
        ) = get_kpsv_eri_pao_projected(
            self.osv,
            self.eris,
            self.pao,
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
