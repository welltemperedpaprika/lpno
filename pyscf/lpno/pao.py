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

import numpy as np
from functools import reduce

from pyscf.scf.addons import canonical_orth_
from pyscf import lib

from pyscf.lpno import tools as dutil

einsum = lib.einsum


def pao(mol, mos, s1e=None, norm_thr=1e-4):
    """Compute PAOs.

    Parameters
    ----------
    norm_thr : float, default=1e-4
        PAOs with norms smaller than ``norm_thr`` are discarded.

    Returns
    -------
    paos : Normalized PAOs.
    inv_idx : Indices mapping AOs to PAOs.

    Notes
    -----
    ``mos`` need to be orthonormal.
    """
    if s1e is None:
        s1e = mol.intor_symmetric('int1e_ovlp')

    mos = np.asarray(mos)
    if mos.ndim == 1:
        mos = mos.reshape(-1, 1)
    assert mos.ndim == 2
    nao, nmo = mos.shape

    paos = np.eye(nao) - mos @ (mos.T.conj() @ s1e)
    norm = np.sqrt(einsum('ui,uv,vi->i', paos.conj(), s1e, paos))
    pao_idx = np.where(norm > norm_thr)[0]
    inv_idx = np.full(nao, -1, dtype=np.int32)
    inv_idx[pao_idx] = np.arange(len(pao_idx))
    paos = paos[:, pao_idx] / norm[pao_idx]
    return paos, inv_idx


def pao_by_atom(mol, paos, atmlst, ao2pao_map=None):
    if ao2pao_map is None:
        ao2pao_map = np.arange(mol.nao)

    aoslices = mol.aoslice_by_atom()[:, 2:]
    pao_idx = np.empty((0,), dtype=np.int32)
    for i0, i1 in aoslices[atmlst].reshape(-1, 2):
        idx = ao2pao_map[i0:i1]
        pao_idx = np.append(pao_idx, idx[idx >= 0])
    return paos[:, pao_idx].reshape(-1, pao_idx.size)


def pao_overlap_with_domain(
        mol, paos, bp_domain, p_domain=None,
        ao2pao_map=None, s1e=None, ovlp_thr=1e-4, orth_thr=1e-6):  # noqa: E125
    """PAOs in the larger domain that overlap with the smaller domain.
    """
    if p_domain is None:
        p_domain = bp_domain
    if s1e is None:
        s1e = mol.intor_symmetric('int1e_ovlp')

    pao_pd = pao_by_atom(mol, paos, p_domain, ao2pao_map)

    x = canonical_orth_(pao_pd.T.conj() @ s1e @ pao_pd, thr=orth_thr)
    pao_pd_orth = pao_pd @ x

    ao_idx_bp = dutil.ao_index_by_atom(mol, bp_domain)
    s21 = s1e[ao_idx_bp]
    s22 = s1e[np.ix_(ao_idx_bp, ao_idx_bp)]
    s22_inv = np.linalg.inv(s22)

    tmp = s21 @ pao_pd_orth
    ovlp = tmp.T.conj() @ s22_inv @ tmp
    w, v = np.linalg.eigh(ovlp)
    return pao_pd_orth @ v[:, w > ovlp_thr]


def get_pao_idxs(mol, domains, ao2pao_map):
    pao2atm_map = []
    for p0, p1 in mol.aoslice_by_atom()[:, 2:]:
        x = ao2pao_map[p0:p1]
        pao2atm_map.append(x[x >= 0])
    pao2atm_map = dutil.list_to_array(pao2atm_map)
    pao_idxs = [reduce(np.union1d, pao2atm_map[domain]) for domain in domains]
    return dutil.list_to_array(pao_idxs)


class PAO:
    def __init__(self, mol, occ_coeff, vir_coeff, vir_energy, domains=None, s1e=None):
        self.thresh_paonorm = 1e-4
        self.s1e = s1e if s1e is not None else mol.intor_symmetric('int1e_ovlp')

        self.domains = domains

        self._build(mol, occ_coeff, vir_coeff, vir_energy)

    def _build(self, mol, occ_coeff, vir_coeff, vir_energy):
        nocc = len(self.domains)
        pao_coeff, ao2pao_map = pao(mol, occ_coeff, s1e=self.s1e, norm_thr=self.thresh_paonorm)
        pao2vir_coeff = reduce(lib.dot, (vir_coeff.T.conj(), self.s1e, pao_coeff))
        pao_ovlp = lib.dot(pao2vir_coeff.T.conj(), pao2vir_coeff)
        pao_fock = lib.dot(pao2vir_coeff.T.conj()*vir_energy, pao2vir_coeff)
        pao2atm_map = []
        for p0, p1 in mol.aoslice_by_atom()[:, 2:]:
            x = ao2pao_map[p0:p1]
            pao2atm_map.append(x[x >= 0])
        pao2atm_map = dutil.list_to_array(pao2atm_map)
        pao_idxs = [reduce(np.union1d, pao2atm_map[self.domains[i]]) for i in range(nocc)]

        self.pao_coeff = pao_coeff
        self.pao2vir_coeff = pao2vir_coeff
        self.pao_fock = pao_fock
        self.pao_ovlp = pao_ovlp
        self.pao2atm_map = pao2atm_map
        self.pao_idxs = pao_idxs

    def get_pao_idx(self, i):
        return self.pao_idxs[i]

    def get_pao_coeff(self, i):
        return self.pao_coeff[:, self.get_pao_idx(i)]

    def get_pao2vir_coeff(self, i):
        return self.pao2vir_coeff[:, self.get_pao_idx(i)]

    def get_pao_ovlp(self, i, j):
        return self.pao_ovlp[self.get_pao_idx(i)][:, self.get_pao_idx(j)]

    def get_pao_fock(self, i, j):
        return self.pao_fock[self.get_pao_idx(i)][:, self.get_pao_idx(j)]
