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

import h5py
import numpy as np
import ctypes
from functools import reduce
from pyscf import lib
from pyscf.lib import logger

from pyscf.lpno import tools
from pyscf.lpno.tools import zdotCNtoR

try:
    libmp = lib.load_library('libmp')
except Exception:
    libmp = None


def _get_cell_permutation(kmesh, T, nao_per_cell):
    '''AO index permutation for cell translation by T lattice vectors.

    Returns perm such that C_shifted = C[perm, :] gives the AO coefficients
    translated by T unit cells.
    '''
    kmesh = np.asarray(kmesh)
    num_cells = int(np.prod(kmesh))
    grid_coords = lib.cartesian_prod([np.arange(k) for k in kmesh])
    shift_vec = grid_coords[T]
    dst_coords = (grid_coords + shift_vec) % kmesh
    strides = np.array([int(np.prod(kmesh[i+1:])) for i in range(len(kmesh))])
    linear_dst = np.dot(dst_coords, strides)
    perm = np.empty(num_cells * nao_per_cell, dtype=int)
    for src in range(num_cells):
        dst = linear_dst[src]
        perm[src * nao_per_cell:(src + 1) * nao_per_cell] = \
            np.arange(dst * nao_per_cell, (dst + 1) * nao_per_cell)
    return perm


def get_osv(eo, ev, dfobj, Tocc=None, Pocc=None, use_evT=True, nlo_per_cell=None,
            **kwargs):
    '''Build pseudocanonicalized OSVs via occupation number thresholding.

    Returns (eosv, uosv) where eosv[i] are OSV energies and uosv[i] are
    vir-to-OSV transformation matrices. If nlo_per_cell is set, builds only
    for ref-cell orbitals and translates via S_T.
    '''
    nocc = dfobj.nocc
    nvir = ev[0].shape[0]
    eosv = []
    uosv = []
    is_pbc = hasattr(dfobj, 'kpts')

    if is_pbc and nlo_per_cell is not None and nlo_per_cell < nocc:
        npc = nlo_per_cell
        nkpts = nocc // npc
        ref_eosv = []
        ref_uosv = []
        for i in range(npc):
            ivLR, ivLI = dfobj.get_occ_blk(i, i + 1)
            ivLR, ivLI = ivLR[0], ivLI[0]
            vii = np.empty((nvir, nvir))
            zdotCNtoR(ivLR, ivLI, ivLR.T, ivLI.T, cR=vii)
            e1, u1 = get_osv1(eo[i], ev[i], vii, Tocc, Pocc)
            vii = None
            ref_eosv.append(e1)
            ref_uosv.append(u1)
        for i_ref in range(npc):
            eosv.append(ref_eosv[i_ref])
            uosv.append(ref_uosv[i_ref])
        vir_coeff = kwargs.get('vir_coeff', None)
        s_ao = kwargs.get('s_ao', None)
        kmesh = kwargs.get('kmesh', None)
        if vir_coeff is not None and s_ao is not None and kmesh is not None:
            nao_pc = vir_coeff.shape[0] // nkpts
            re_pseudocano = kwargs.get('re_pseudocano', False)
            S_C = lib.dot(s_ao, vir_coeff)
            for T in range(1, nkpts):
                perm = _get_cell_permutation(kmesh, T, nao_pc)
                # S_T = C_vir^T @ S_ao @ C_vir[perm_T],  shape (Nvir, Nvir)
                # The correct OSV translation is u[j] = S_T^T @ u[j_ref]
                # because vii[j] = S_T^T @ vii[j_ref] @ S_T
                S_T = lib.dot(vir_coeff.T, S_C[perm])
                for i_ref in range(npc):
                    u_translated = lib.dot(S_T.T, ref_uosv[i_ref])
                    if re_pseudocano:
                        e_pc, u_pc = tools.pseudocano(ev[i_ref], u_translated)
                        eosv.append(e_pc)
                        uosv.append(u_pc)
                    else:
                        # ref_uosv is already pseudocanonicalized (from get_osv1).
                        # S_T commutes with diag(ev) to ~1e-12, so the Fock in
                        # the translated basis is diagonal to machine precision.
                        eosv.append(ref_eosv[i_ref])
                        uosv.append(u_translated)
        else:
            for R in range(1, nkpts):
                for i_ref in range(npc):
                    eosv.append(ref_eosv[i_ref])
                    uosv.append(ref_uosv[i_ref])
    else:
        for i in range(nocc):
            if is_pbc:
                ivLR, ivLI = dfobj.get_occ_blk(i, i + 1)
                ivLR, ivLI = ivLR[0], ivLI[0]
                vii = np.empty((nvir, nvir))
                zdotCNtoR(ivLR, ivLI, ivLR.T, ivLI.T, cR=vii)
            else:
                ivL = dfobj.get_occ_blk(i, i+1)[0]
                vii = lib.dot(ivL, ivL.T)
            e1, u1 = get_osv1(eo[i], ev[i], vii, Tocc, Pocc)
            ivL = vii = None
            eosv.append(e1)
            uosv.append(u1)

    return eosv, uosv


def get_osv1(eo, ev, vii, Tocc=None, Pocc=None, use_evT=True):
    evv = ev[:, None] + ev
    tii = vii.conj() / (eo*2 - evv)
    n, v = np.linalg.eigh(tii)
    vact = select_osv1(n, v, Tocc, Pocc, use_evT)
    eact, vact = tools.pseudocano(ev, vact)
    return eact, vact


def select_osv1(n, v, Tocc=None, Pocc=None, use_evT=True):
    if not use_evT:
        n = n**2.   # eigval of 1rdm
    vact = tools.select_natorb(n, v, thresh=Tocc, pct_occ=Pocc)
    return vact


def get_osv_ovlp(osv, pair_mask=None, s=None):
    nocc = osv.nocc
    nosv = osv.nosv
    if pair_mask is None:
        pair_mask = np.ones((nocc, nocc), dtype=bool)
    pair_shape_full = [(nosv[i], nosv[j]) if pair_mask[i, j] else (0, 0)
                       for i in range(nocc) for j in range(i+1)]
    S = tools.RaggedTensor2DTril(nocc, pair_shape_full, True)
    if s is None:
        get_Sij = lambda u1T, u2: lib.dot(u1T, u2)
    else:
        get_Sij = lambda u1T, u2: reduce(lib.dot, (u1T, s, u2))
    for i in range(nocc):
        uiT = osv.u[i].T.conj()
        for j in range(i):
            if not pair_mask[i, j]:
                continue
            S[i, j] = get_Sij(uiT, osv.u[j])
        S[i, i] = np.eye(nosv[i])
    return S


def get_osv_fock(osv, vir_energy, pair_mask=None):
    nocc = osv.nocc
    nosv = osv.nosv
    if pair_mask is None:
        pair_mask = np.ones((nocc, nocc), dtype=bool)
    pair_shape_full = [(nosv[i], nosv[j]) if pair_mask[i, j] else (0, 0)
                       for i in range(nocc) for j in range(i+1)]
    F = tools.RaggedTensor2DTril(nocc, pair_shape_full, True)
    get_Fij = lambda u1T, u2: lib.dot(u1T*vir_energy, u2)
    for i in range(nocc):
        uiT = osv.u[i].T.conj()
        for j in range(i):
            if not pair_mask[i, j]:
                continue
            F[i, j] = get_Fij(uiT, osv.u[j])
        F[i, i] = np.diag(osv.e[i])
    return F


class OSV:
    '''Orbital-specific virtual (OSV) data container.

    Attributes: nocc, nvir, nosv (list), u (RaggedTensor1D of vir-to-OSV
    transforms), e (RaggedTensor1D of OSV energies), S/F (overlap/Fock
    between OSV pairs).
    '''
    def __init__(self, osv_param, eris, occ_energy, vir_energy, verbose=None):
        log = logger.new_logger(eris, verbose)

        self.vir_energy = vir_energy

        e, u = self._build(eris, occ_energy, vir_energy, osv_param, log)

        self.nocc = len(u)
        self.nvir = u[0].shape[0]
        self.nosv = [x.shape[1] for x in u]
        self.u = tools.RaggedTensor1D().init_from_data(u)
        self.e = tools.RaggedTensor1D().init_from_data(e)

        self.S = None
        self.F = None

    def _build(self, eris, occ_energy, vir_energy, osv_param, log):
        param = {
            'Tocc': osv_param.get('thresh', None),
            'Pocc': osv_param.get('pct_occ', None),
            'use_evT': osv_param.get('use_evT', True),
        }
        nlo_per_cell = osv_param.get('nlo_per_cell', None)
        vir_coeff = osv_param.get('vir_coeff', None)
        s_ao = osv_param.get('s_ao', None)
        kmesh = osv_param.get('kmesh', None)
        re_pseudocano = osv_param.get('re_pseudocano', False)
        log.info('Generating OSV with parameters: %s', param)
        e, u = get_osv(occ_energy, [vir_energy]*eris.nocc, eris,
                       nlo_per_cell=nlo_per_cell,
                       vir_coeff=vir_coeff, s_ao=s_ao, kmesh=kmesh,
                       re_pseudocano=re_pseudocano,
                       **param)
        return e, u

    def get_ovlp(self, pair_mask=None, force_update=False):
        if self.S is None or force_update:
            self.S = get_osv_ovlp(self, pair_mask)
        return self.S

    def get_fock(self, pair_mask=None, force_update=False):
        if self.F is None or force_update:
            self.F = get_osv_fock(self, self.vir_energy, pair_mask)
        return self.F


class OSV_C(OSV):
    def _build(self, eris, occ_energy, vir_energy, osv_param, log):
        param = {
            'Tocc': osv_param.get('thresh', None),
            'Pocc': osv_param.get('pct_occ', None),
            'use_evT': osv_param.get('use_evT', True),
        }

        nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux

        evv = np.asarray(vir_energy[:, None] + vir_energy, order='C')

        nvir2 = nvir**2
        mem_avail = eris.max_memory - lib.current_memory()[0]
        mem_blk = (nvir2+nvir*naux) * 8/1e6
        occ_blksize = max(1, min(nocc, int(np.floor(mem_avail * 0.5 / mem_blk))))
        log.debug('occ_blksize for make_osv: %d', occ_blksize)

        tspans = np.zeros((3, 2))
        tnames = ['load', 'tii-eigval', 'pseudocano']

        drv = libmp.get_osv

        es = []
        us = []
        buf_n = np.empty(occ_blksize*nvir, dtype=np.float64)
        buf_v = np.empty(occ_blksize*nvir2, dtype=eris.dtype)
        for i0, i1 in lib.prange(0, nocc, occ_blksize):
            TICK = np.asarray((logger.process_clock(), logger.perf_counter()))
            nocci = i1 - i0
            ivL = eris.get_occ_blk(i0, i1)
            TOCK = np.asarray((logger.process_clock(), logger.perf_counter()))
            tspans[0] += TOCK - TICK

            drv(
                buf_n.ctypes.data_as(ctypes.c_void_p),
                buf_v.ctypes.data_as(ctypes.c_void_p),
                ivL.ctypes.data_as(ctypes.c_void_p),
                np.asarray(occ_energy[i0:i1], order='C').ctypes.data_as(ctypes.c_void_p),
                evv.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nocci), ctypes.c_int(nvir), ctypes.c_int(naux)
            )

            TICK = np.asarray((logger.process_clock(), logger.perf_counter()))
            tspans[1] += TICK - TOCK

            for i in range(i0, i1):
                ii = i - i0
                n = buf_n[ii*nvir:(ii+1)*nvir]
                v = buf_v[ii*nvir2:(ii+1)*nvir2].reshape(nvir, nvir).T
                u = select_osv1(n, v, **param)
                e, u = tools.pseudocano(vir_energy, u)
                es.append(e)
                us.append(u)

            TOCK = np.asarray((logger.process_clock(), logger.perf_counter()))
            tspans[2] += TOCK - TICK
        buf_n = buf_v = None

        for tspan, tname in zip(tspans, tnames):
            log.info('make_osv CPU time for %-10s  %9.2f sec  wall time %9.2f sec', tname, *tspan)
        log.info('')

        return es, us

    def get_ovlp(self, pair_mask=None, force_update=False):
        if self.S is None or force_update:
            nocc = self.nocc
            nosv = self.nosv
            nvir = self.u[0].shape[0]
            if pair_mask is None:
                pair_mask = np.ones((nocc, nocc), dtype=bool)
            pair_shape_full = [(nosv[i], nosv[j]) if pair_mask[i, j] else (0, 0)
                               for i in range(nocc) for j in range(i+1)]
            S = tools.RaggedTensor2DTril(nocc, pair_shape_full, True)
            drv = libmp.get_OSV_S

            drv(
                S.data.ctypes.data_as(ctypes.c_void_p),
                np.asarray(S.loc, dtype=np.int32).ctypes.data_as(ctypes.c_void_p),
                np.asarray(pair_mask, dtype=np.int8, order='C').ctypes.data_as(ctypes.c_void_p),
                self.u.data.ctypes.data_as(ctypes.c_void_p),
                np.asarray(self.u.loc, dtype=np.int32).ctypes.data_as(ctypes.c_void_p),
                np.asarray(nosv, dtype=np.int32).ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nocc), ctypes.c_int(nvir)
            )

            self.S = S
        return self.S

    def get_fock(self, pair_mask=None, force_update=False):
        if self.F is None or force_update:
            nocc = self.nocc
            nosv = self.nosv
            nvir = self.u[0].shape[0]
            if pair_mask is None:
                pair_mask = np.ones((nocc, nocc), dtype=bool)
            pair_shape_full = [(nosv[i], nosv[j]) if pair_mask[i, j] else (0, 0)
                               for i in range(nocc) for j in range(i+1)]
            F = tools.RaggedTensor2DTril(nocc, pair_shape_full, True)
            drv = libmp.get_OSV_F

            drv(
                F.data.ctypes.data_as(ctypes.c_void_p),
                np.asarray(F.loc, dtype=np.int32).ctypes.data_as(ctypes.c_void_p),
                np.asarray(pair_mask, dtype=np.int8, order='C').ctypes.data_as(ctypes.c_void_p),
                self.vir_energy.ctypes.data_as(ctypes.c_void_p),
                self.u.data.ctypes.data_as(ctypes.c_void_p),
                np.asarray(self.u.loc, dtype=np.int32).ctypes.data_as(ctypes.c_void_p),
                np.asarray(nosv, dtype=np.int32).ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nocc), ctypes.c_int(nvir)
            )

            self.F = F
        return self.F
