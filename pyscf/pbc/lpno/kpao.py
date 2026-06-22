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

"""Periodic PAO construction and translated domain helpers."""

from functools import reduce

import numpy as np
from pyscf import lib
from pyscf.lib import logger

from pyscf.lpno import domain as domain_mod
from pyscf.lpno import osv as osv_mod
from pyscf.lpno import pao as pao_mod
from pyscf.pbc.lpno.kpair_domain import PeriodicPairIndex
from pyscf.lpno.tools import RaggedTensor1D, RaggedTensor2DTril, safe_eigh, zdotCNtoR


def _stable_unique(values):
    """Return unique integer values without changing their first-seen order."""
    values = np.asarray(values, dtype=np.int32).ravel()
    if values.size == 0:
        return values
    seen = set()
    out = []
    for value in values:
        ivalue = int(value)
        if ivalue in seen:
            continue
        seen.add(ivalue)
        out.append(ivalue)
    return np.asarray(out, dtype=np.int32)


def _domain_size_stats(sizes):
    sizes = np.asarray(sizes, dtype=np.int32).ravel()
    if sizes.size == 0:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "avg": 0.0,
        }
    return {
        "count": int(sizes.size),
        "min": int(np.min(sizes)),
        "max": int(np.max(sizes)),
        "avg": float(np.mean(sizes)),
    }


def reference_atom_domains(
    mol,
    occ_coeff,
    nlo_per_cell,
    mode="full",
    s1e=None,
    bp_thr=0.999,
    q_thr=None,
    principal_thr=0.99,
):
    """Build reference-cell atom domains for periodic PAO construction."""
    mode = "full" if mode is None else str(mode).lower()
    nlo_per_cell = int(nlo_per_cell)
    if nlo_per_cell <= 0:
        raise ValueError("nlo_per_cell must be positive")
    if np.asarray(occ_coeff).shape[1] < nlo_per_cell:
        raise ValueError("occ_coeff must include every reference-cell orbital")

    if mode == "full":
        full_domain = np.arange(mol.natm, dtype=np.int32)
        return [full_domain.copy() for _ in range(nlo_per_cell)]

    if mode == "bp":
        domains = domain_mod.get_bp_domain(
            mol,
            np.asarray(occ_coeff)[:, :nlo_per_cell],
            s1e=s1e,
            bp_thr=bp_thr,
            q_thr=q_thr,
        )
        return [_stable_unique(domain) for domain in domains]

    if mode in ("principal", "doi"):
        return _principal_atom_domains(
            mol,
            np.asarray(occ_coeff)[:, :nlo_per_cell],
            s1e=s1e,
            principal_thr=principal_thr,
            q_thr=q_thr,
        )

    raise ValueError(f"unknown PAO domain mode {mode!r}")


def _principal_atom_domains(
    mol,
    mos,
    s1e=None,
    principal_thr=0.99,
    q_thr=None,
    atmlst=None,
):
    """Principal atom domains from cumulative occupied Mulliken population."""
    if not (0.0 < float(principal_thr) <= 1.0):
        raise ValueError("principal_thr must be in (0, 1]")
    if q_thr is not None and not (0.0 <= float(q_thr) <= 1.0):
        raise ValueError("q_thr must be in [0, 1] for principal domains")
    if s1e is None:
        s1e = mol.intor_symmetric("int1e_ovlp")
    if atmlst is None:
        atmlst = np.arange(mol.natm, dtype=np.int32)
    else:
        atmlst = np.asarray(atmlst, dtype=np.int32)

    mos = np.asarray(mos)
    if mos.ndim == 1:
        mos = mos.reshape(-1, 1)
    if mos.ndim != 2:
        raise ValueError("mos must be a 1D or 2D coefficient array")
    nao, nmo = mos.shape
    if nao != mol.nao:
        raise ValueError("mos row count must match mol.nao")

    aoslices = mol.aoslice_by_atom()[:, 2:]
    domains = []
    eps = np.finfo(float).eps
    for i in range(nmo):
        orb = mos[:, i]
        gross_pop = orb * lib.einsum("pq,q->p", s1e, orb.conj())
        atom_pop = np.abs(np.asarray([
            gross_pop[slice(*aoslices[atom])].sum()
            for atom in atmlst
        ], dtype=np.result_type(gross_pop, float))).real
        total = float(atom_pop.sum())
        if total <= eps:
            domains.append(np.asarray([atmlst[0]], dtype=np.int32))
            continue

        weights = atom_pop / total
        order = np.argsort(-weights, kind="mergesort")
        selected = []
        cumulative = 0.0
        for idx in order:
            keep_for_cumulative = cumulative < principal_thr
            keep_for_floor = q_thr is not None and weights[idx] >= q_thr
            if not keep_for_cumulative and not keep_for_floor:
                continue
            selected.append(int(atmlst[idx]))
            cumulative += float(weights[idx])
        if not selected:
            selected.append(int(atmlst[order[0]]))
        domains.append(np.asarray(selected, dtype=np.int32))
    return domains


class PeriodicPAO:
    """Supercell PAOs with reference-cell translated atom/domain lookup.

    This helper owns PAO construction and index algebra only.  It deliberately
    does not alter OSV/PNO residual equations or production defaults.
    """

    def __init__(
        self,
        mol,
        occ_coeff,
        nlo_per_cell,
        nlo,
        cell_sub,
        vir_coeff=None,
        vir_energy=None,
        ref_atom_domains=None,
        s1e=None,
        norm_thr=1e-4,
    ):
        self.mol = mol
        self.s1e = s1e if s1e is not None else mol.intor_symmetric("int1e_ovlp")
        self.nlo_per_cell = int(nlo_per_cell)
        self.nlo = int(nlo)
        self.pair_index = PeriodicPairIndex(nlo_per_cell, nlo, cell_sub)
        self.ncell = self.pair_index.ncell
        self.norm_thr = norm_thr

        if mol.natm % self.ncell != 0:
            raise ValueError("number of atoms must be a multiple of ncell")
        self.natm_per_cell = mol.natm // self.ncell

        self.pao_coeff, self.ao2pao_map = pao_mod.pao(
            mol, occ_coeff, s1e=self.s1e, norm_thr=norm_thr)
        self.pao_parent_ao = np.where(self.ao2pao_map >= 0)[0]
        self.npao = self.pao_coeff.shape[1]
        self.atom_to_pao = self._build_atom_to_pao()
        self.npao_per_cell = self._count_paos_by_cell()

        self.ref_atom_domains = self._init_ref_atom_domains(ref_atom_domains)
        self.pruned_nao = int(self.s1e.shape[0] - self.npao)

        if vir_coeff is not None:
            self.pao2vir_coeff = reduce(
                lib.dot, (vir_coeff.T.conj(), self.s1e, self.pao_coeff))
            self.pao_ovlp = lib.dot(self.pao2vir_coeff.T.conj(), self.pao2vir_coeff)
            if vir_energy is not None:
                self.pao_fock = lib.dot(
                    self.pao2vir_coeff.T.conj() * vir_energy,
                    self.pao2vir_coeff)
            else:
                self.pao_fock = None
        else:
            self.pao2vir_coeff = None
            self.pao_ovlp = reduce(
                lib.dot, (self.pao_coeff.T.conj(), self.s1e, self.pao_coeff))
            self.pao_fock = None

    def _build_atom_to_pao(self):
        atom_to_pao = []
        for p0, p1 in self.mol.aoslice_by_atom()[:, 2:]:
            pao_idx = self.ao2pao_map[p0:p1]
            atom_to_pao.append(np.asarray(pao_idx[pao_idx >= 0], dtype=np.int32))
        return atom_to_pao

    def _count_paos_by_cell(self):
        counts = np.zeros(self.ncell, dtype=np.int32)
        for atom, pao_idx in enumerate(self.atom_to_pao):
            counts[self.atom_cell(atom)] += len(pao_idx)
        return counts

    def _init_ref_atom_domains(self, ref_atom_domains):
        if ref_atom_domains is None:
            full_domain = np.arange(self.mol.natm, dtype=np.int32)
            return [full_domain.copy() for _ in range(self.nlo_per_cell)]
        if len(ref_atom_domains) != self.nlo_per_cell:
            raise ValueError(
                "ref_atom_domains must have one entry per reference-cell LO")
        return [
            np.asarray(domain, dtype=np.int32).copy()
            for domain in ref_atom_domains
        ]

    def atom_ref_cell(self, atom_idx):
        atom_idx = int(atom_idx)
        if not (0 <= atom_idx < self.mol.natm):
            raise ValueError(
                f"atom index {atom_idx} outside [0, {self.mol.natm})")
        return atom_idx % self.natm_per_cell, atom_idx // self.natm_per_cell

    def atom_cell(self, atom_idx):
        return self.atom_ref_cell(atom_idx)[1]

    def translate_atom(self, atom_idx, target_cell):
        atom_ref, atom_cell = self.atom_ref_cell(atom_idx)
        shifted_cell = self.pair_index.relative_cell(
            atom_cell, self.pair_index.inverse_cell(target_cell))
        return int(shifted_cell * self.natm_per_cell + atom_ref)

    def translated_atom_domain(self, i_ref, target_cell):
        if not (0 <= int(i_ref) < self.nlo_per_cell):
            raise ValueError(
                f"reference LO index {i_ref} outside [0, {self.nlo_per_cell})")
        atoms = [
            self.translate_atom(atom, target_cell)
            for atom in self.ref_atom_domains[int(i_ref)]
        ]
        return _stable_unique(atoms)

    def atom_pao_indices(self, atom_idx):
        return self.atom_to_pao[int(atom_idx)]

    def pao_domain_from_atoms(self, atom_domain):
        chunks = [self.atom_pao_indices(atom) for atom in atom_domain]
        chunks = [chunk for chunk in chunks if len(chunk)]
        if not chunks:
            return np.zeros(0, dtype=np.int32)
        return _stable_unique(np.concatenate(chunks))

    def translated_pao_domain(self, i_ref, target_cell):
        atoms = self.translated_atom_domain(i_ref, target_cell)
        return self.pao_domain_from_atoms(atoms)

    def orbital_pao_domain(self, global_lo_idx):
        lo_ref, cell_idx = self.pair_index.ref_cell(global_lo_idx)
        return self.translated_pao_domain(lo_ref, cell_idx)

    def pair_pao_domain(self, i_ref, j_global):
        domain_i = self.translated_pao_domain(i_ref, 0)
        domain_j = self.orbital_pao_domain(j_global)
        return _stable_unique(np.concatenate((domain_i, domain_j)))

    def orbital_domain_sizes(self):
        return np.asarray([
            len(self.translated_pao_domain(i_ref, 0))
            for i_ref in range(self.nlo_per_cell)
        ], dtype=np.int32)

    def pair_domain_sizes(self, pair_mask=None):
        if pair_mask is None:
            pair_mask = np.ones((self.nlo_per_cell, self.nlo), dtype=bool)
        else:
            pair_mask = self.pair_index.validate_ref_pair_mask(
                pair_mask, label="PAO pair domain").copy()
        sizes = [
            len(self.pair_pao_domain(i_ref, j_global))
            for i_ref in range(self.nlo_per_cell)
            for j_global in range(self.nlo)
            if pair_mask[i_ref, j_global]
        ]
        return np.asarray(sizes, dtype=np.int32)

    def pair_domain_size_summary(self, pair_mask=None):
        return _domain_size_stats(self.pair_domain_sizes(pair_mask))

    def get_pao_idx(self, global_lo_idx):
        """Return the PAO domain for a global localized occupied orbital."""
        return self.orbital_pao_domain(global_lo_idx)

    def get_pao_coeff(self, pao_idx=None):
        if pao_idx is None:
            return self.pao_coeff
        return self.pao_coeff[:, pao_idx]

    def get_orbital_pao_coeff(self, global_lo_idx):
        return self.get_pao_coeff(self.get_pao_idx(global_lo_idx))

    def get_pao2vir_coeff(self, pao_idx=None):
        if self.pao2vir_coeff is None:
            raise RuntimeError("pao2vir_coeff was not built without vir_coeff")
        if pao_idx is None:
            return self.pao2vir_coeff
        return self.pao2vir_coeff[:, pao_idx]

    def get_orbital_pao2vir_coeff(self, global_lo_idx):
        return self.get_pao2vir_coeff(self.get_pao_idx(global_lo_idx))

    def get_pair_pao2vir_coeff(self, i_ref, j_global):
        return self.get_pao2vir_coeff(self.pair_pao_domain(i_ref, j_global))

    def get_pao_ovlp(self, pao_idx_i, pao_idx_j=None):
        if pao_idx_j is None:
            pao_idx_j = pao_idx_i
        return self.pao_ovlp[np.ix_(pao_idx_i, pao_idx_j)]

    def get_orbital_pao_ovlp(self, i_global, j_global=None):
        idx_i = self.get_pao_idx(i_global)
        idx_j = idx_i if j_global is None else self.get_pao_idx(j_global)
        return self.get_pao_ovlp(idx_i, idx_j)

    def get_pao_fock(self, pao_idx_i, pao_idx_j=None):
        if self.pao_fock is None:
            raise RuntimeError("pao_fock was not built without vir_energy")
        if pao_idx_j is None:
            pao_idx_j = pao_idx_i
        return self.pao_fock[np.ix_(pao_idx_i, pao_idx_j)]

    def get_orbital_pao_fock(self, i_global, j_global=None):
        idx_i = self.get_pao_idx(i_global)
        idx_j = idx_i if j_global is None else self.get_pao_idx(j_global)
        return self.get_pao_fock(idx_i, idx_j)

    def domain_summary(self):
        domain_sizes = self.orbital_domain_sizes()
        return {
            "ncell": self.ncell,
            "natm_per_cell": self.natm_per_cell,
            "npao": self.npao,
            "pruned_nao": self.pruned_nao,
            "npao_per_cell": self.npao_per_cell.copy(),
            "ref_domain_sizes": domain_sizes,
            "pair_domain_size_summary": self.pair_domain_size_summary(),
        }


class PeriodicPAOBasisOSV:
    """Periodic OSVs whose coefficient rows are local PAO-domain functions."""

    def __init__(self, eris, occ_energy, pao, osv_param=None, verbose=None):
        if osv_param is None:
            osv_param = {}
        self.pao = pao
        self.occ_energy = np.asarray(occ_energy)
        self.nlo_per_cell = int(osv_param.get(
            "nlo_per_cell", getattr(pao, "nlo_per_cell", 0)))
        self.nocc = int(osv_param.get("nlo", getattr(pao, "nlo", 0)))
        if self.nlo_per_cell <= 0 or self.nocc <= 0:
            raise ValueError("nlo_per_cell and nlo must be positive")
        if self.nocc % self.nlo_per_cell != 0:
            raise ValueError("nlo must be a multiple of nlo_per_cell")
        if self.occ_energy.shape[0] < self.nocc:
            raise ValueError("occ_energy must include every occupied orbital")

        self.ncell = self.nocc // self.nlo_per_cell
        self.lindep_thr = osv_param.get("lindep_thr", 1e-6)
        log = logger.new_logger(eris, verbose)
        e, u = self._build(eris, osv_param, log)

        self.nosv = [x.shape[1] for x in u]
        self.npao_domain = [x.shape[0] for x in u]
        self.u = RaggedTensor1D().init_from_data(u)
        self.e = RaggedTensor1D().init_from_data(e)
        self.S = None
        self.F = None

    def _build(self, eris, osv_param, log):
        param = {
            "Tocc": osv_param.get("thresh", None),
            "Pocc": osv_param.get("pct_occ", None),
            "use_evT": osv_param.get("use_evT", True),
        }
        ref_es = []
        ref_us = []
        log.info(
            "Building PAO-basis OSVs for %d reference-cell orbitals",
            self.nlo_per_cell,
        )
        for i_ref in range(self.nlo_per_cell):
            epao, xcano = safe_eigh(
                self.pao.get_orbital_pao_fock(i_ref),
                self.pao.get_orbital_pao_ovlp(i_ref),
                self.lindep_thr,
            )
            pao2vir = self.pao.get_orbital_pao2vir_coeff(i_ref)
            if pao2vir.shape[1] != xcano.shape[0]:
                raise ValueError(
                    "PAO projector/domain size mismatch for orbital "
                    f"{i_ref}: projector has {pao2vir.shape[1]} PAOs, "
                    f"orthogonalizer has {xcano.shape[0]} rows"
                )
            profile_start = None
            if hasattr(eris, "record_projector_formation") and getattr(
                eris, "_pbc_profile_enabled", False
            ):
                profile_start = logger.process_clock(), logger.perf_counter()
            projector = lib.dot(pao2vir, xcano)
            if profile_start is not None:
                cpu0, wall0 = profile_start
                eris.record_projector_formation(
                    "pao_pcpao",
                    rows=projector.shape[0],
                    cols=projector.shape[1],
                    cpu=logger.process_clock() - cpu0,
                    wall=logger.perf_counter() - wall0,
                )
            ivLR, ivLI = eris.get_projected_blk(i_ref, projector)
            vii = np.empty((xcano.shape[1], xcano.shape[1]))
            zdotCNtoR(ivLR, ivLI, ivLR.T, ivLI.T, cR=vii)

            e1, u1_pcpao = osv_mod.get_osv1(
                self.occ_energy[i_ref], epao, vii, **param)
            ref_es.append(e1)
            ref_us.append(lib.dot(xcano, u1_pcpao))

        es = []
        us = []
        for _cell in range(self.ncell):
            for i_ref in range(self.nlo_per_cell):
                es.append(ref_es[i_ref].copy())
                us.append(ref_us[i_ref].copy())
        return es, us

    def get_canonical_projector(self, global_lo_idx):
        """Return canonical-virtual columns for this PAO-basis OSV domain."""
        return lib.dot(
            self.pao.get_orbital_pao2vir_coeff(global_lo_idx),
            self.u[global_lo_idx],
        )

    def get_ovlp(self, pair_mask=None, force_update=False):
        if self.S is not None and not force_update:
            return self.S
        if pair_mask is None:
            pair_mask = np.ones((self.nocc, self.nocc), dtype=bool)
        else:
            pair_mask = np.asarray(pair_mask, dtype=bool)
            if pair_mask.shape != (self.nocc, self.nocc):
                raise ValueError(
                    f"pair_mask shape must be {(self.nocc, self.nocc)}, "
                    f"got {pair_mask.shape}"
                )

        shape = [
            (self.nosv[i], self.nosv[j]) if pair_mask[i, j] else (0, 0)
            for i in range(self.nocc)
            for j in range(i + 1)
        ]
        s_osv = RaggedTensor2DTril(self.nocc, shape, True)
        for i in range(self.nocc):
            s_osv[i, i] = np.eye(self.nosv[i])
            uiT = self.u[i].T.conj()
            for j in range(i):
                if not pair_mask[i, j]:
                    continue
                s_pao = self.pao.get_orbital_pao_ovlp(i, j)
                s_osv[i, j] = reduce(lib.dot, (uiT, s_pao, self.u[j]))
        self.S = s_osv
        return self.S

    def get_fock(self, pair_mask=None, force_update=False):
        if self.F is not None and not force_update:
            return self.F
        if pair_mask is None:
            pair_mask = np.ones((self.nocc, self.nocc), dtype=bool)
        else:
            pair_mask = np.asarray(pair_mask, dtype=bool)
            if pair_mask.shape != (self.nocc, self.nocc):
                raise ValueError(
                    f"pair_mask shape must be {(self.nocc, self.nocc)}, "
                    f"got {pair_mask.shape}"
                )

        shape = [
            (self.nosv[i], self.nosv[j]) if pair_mask[i, j] else (0, 0)
            for i in range(self.nocc)
            for j in range(i + 1)
        ]
        f_osv = RaggedTensor2DTril(self.nocc, shape, True)
        for i in range(self.nocc):
            f_osv[i, i] = np.diag(self.e[i])
            uiT = self.u[i].T.conj()
            for j in range(i):
                if not pair_mask[i, j]:
                    continue
                f_pao = self.pao.get_orbital_pao_fock(i, j)
                f_osv[i, j] = reduce(lib.dot, (uiT, f_pao, self.u[j]))
        self.F = f_osv
        return self.F
