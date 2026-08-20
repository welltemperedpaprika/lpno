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

from pyscf.lpno import tools as util

from pyscf import lib

einsum = lib.einsum


def get_mkn_domain(mol, mos, s1e=None, q_thr=1e-2, atmlst=None):
    ''' Domain by Mulliken populations
    '''
    if s1e is None:
        s1e = mol.intor_symmetric('int1e_ovlp')
    if atmlst is None:
        atmlst = np.arange(mol.natm)

    mos = np.asarray(mos)
    if mos.ndim == 1:
        mos = mos.reshape(-1, 1)
    assert mos.ndim == 2
    nao, nmo = mos.shape
    # TODO project MOs onto smaller basis
    assert nao == mol.nao

    aoslices = mol.aoslice_by_atom()[:, 2:]
    bp_atmlst = []

    for i in range(nmo):
        orbi = mos[:, i]
        GOP = orbi * einsum('pq,q->p', s1e, orbi.conj())

        # FIXME Mulliken charge can be negative
        q = abs(np.asarray([GOP[slice(*aoslices[a])].sum() for a in atmlst]))

        bp_atmlst.append(atmlst[q > q_thr])

    return util.list_to_array(bp_atmlst)


def get_bp_domain(mol, mos, s1e=None, bp_thr=0.999,
                  q_thr=None, atmlst=None):
    """BP domains based on partial Mulliken charges.
    """
    if s1e is None:
        if hasattr(mol, 'pbc_intor'):
            s1e = mol.pbc_intor('int1e_ovlp')
        else:
            s1e = mol.intor_symmetric('int1e_ovlp')
    if q_thr is None:
        q_thr = min(0.05, 5*(1-bp_thr))
    if atmlst is None:
        atmlst = np.arange(mol.natm)

    mos = np.asarray(mos)
    if mos.ndim == 1:
        mos = mos.reshape(-1, 1)
    assert mos.ndim == 2
    nao, nmo = mos.shape
    # TODO project MOs onto smaller basis
    assert nao == mol.nao

    rr = atom_distance(mol, atmlst)
    aoslices = mol.aoslice_by_atom()[:, 2:]
    bp_atmlst = []

    for i in range(nmo):
        orbi = mos[:, i]
        GOP = orbi * einsum('pq,q->p', s1e, orbi.conj())

        # FIXME Mulliken charge can be negative
        q = abs(np.asarray([GOP[slice(*aoslices[a])].sum() for a in atmlst]))

        _atms = atmlst[q > q_thr]
        av = _compute_av(mol, orbi, s1e, _atms)

        if av < bp_thr:
            center_id = np.argsort(-q)[0]
            _sorted_atm_idx = np.argsort(rr[center_id])

            for iatm in _sorted_atm_idx[1:]:
                a = atmlst[iatm]
                if a not in _atms:
                    _atms = np.append(_atms, a)
                    av = _compute_av(mol, orbi, s1e, _atms)
                    if av >= bp_thr:
                        break
        bp_atmlst.append(_atms)

    return util.list_to_array(bp_atmlst)


def _compute_av(mol, mo, s1e=None, atmlst=None):
    """Compute BP value
    """
    if s1e is None:
        s1e = mol.intor_symmetric('int1e_ovlp')
    if atmlst is None:
        atmlst = np.arange(mol.natm)

    mo = np.asarray(mo)

    ao_idx = util.ao_index_by_atom(mol, atmlst)
    v = s1e[ao_idx] @ mo
    a = np.linalg.solve(s1e[np.ix_(ao_idx, ao_idx)], v)
    av = np.sum(a * v, axis=0)
    return av


def atom_distance(mol, atmlst=None):
    """Atomic distance array
    """
    from pyscf.gto.mole import inter_distance
    if atmlst is None:
        atmlst = np.arange(mol.natm)
    coords = mol.atom_coords()[atmlst].reshape(-1, 3)
    r12 = coords[:, None, :] - coords[None, :, :]
    if hasattr(mol, 'a') and mol.a is not None:
        lat = mol.a
        inv_lat = np.linalg.inv(lat)
        r12_frac = lib.einsum('ijx,xy->ijy', r12, inv_lat.T)
        r12_frac -= np.rint(r12_frac)
        r12 = lib.einsum('ijy,xy->ijx', r12_frac, lat.T)
        return np.linalg.norm(r12, axis=-1)
    return inter_distance(mol, coords=coords)


def get_extended_domain(domains, pair_mask):
    ext_domains = []
    for i, domain in enumerate(domains):
        jdx = np.where(pair_mask[i])[0]
        ext_domains.append(reduce(np.union1d, domains[jdx]))
    return util.list_to_array(ext_domains)


def inverse_domain(domains, n):
    inv_domains = [[] for i in range(n)]
    for i, domain in enumerate(domains):
        for x in domain:
            inv_domains[x].append(i)
    for i in range(n):
        inv_domains[i] = np.unique(inv_domains[i])
    return util.list_to_array(inv_domains)


def concatenate_domain(domains1, domains2):
    return util.list_to_array([reduce(np.union1d, domains2[domain1]) for domain1 in domains1])
