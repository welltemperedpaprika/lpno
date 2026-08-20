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

import ctypes
import numpy as np
from functools import reduce

from pyscf import lib
from pyscf.lib import logger

from pyscf.lpno import tools
from pyscf.lpno.pno import get_pno1

einsum = lib.einsum

try:
    libmp = lib.load_library('libmp')
except Exception:
    libmp = None


def get_scmp2_energy(moeocc, moevir, K_dosv, pair_mask):
    nocc = len(moeocc)
    pair_mask = tools._to_full(pair_mask, nocc)

    epair = np.zeros((nocc, nocc))
    for i in range(nocc):
        for j in range(i+1):
            if not pair_mask[i, j]:
                continue
            Kij = K_dosv[i, j]
            Tij = Kij.conj() / (moeocc[i]+moeocc[j]-(moevir[i][:, None]+moevir[j]))
            epair[i, j] = einsum('ab,ab->', Kij, Tij)
    epair = lib.pack_tril(epair)
    pair_mask = tools._to_tril(pair_mask, nocc)

    return epair


def get_dosv_eri(osv, eris, pair_mask, with_K=True, with_J=True, max_memory=None):
    ''' Calculate dOSV-type ERIs, i.e.,
            K[i,j](a_i, b_j) = (i a_i | j b_j)
            J[i,j](a_i, b_j) = (j a_i | i b_i)

        Args:
            with_K/J:
                If set to False, None will be returned for K/J

        Returns:
            K, J
    '''
    log = logger.new_logger(eris)

    if max_memory is None:
        max_memory = eris.max_memory

    dtype = np.float64
    dsize = 8

    nocc = osv.nocc
    nosv = osv.nosv
    pair_mask = tools._to_full(pair_mask, nocc)

    if not (with_J or with_K):
        return None, None

    pair_shape = [(nosv[i], nosv[j]) if pair_mask[i, j] else (0, 0)
                  for i in range(nocc) for j in range(i+1)]
    mem_est = sum([np.prod(x) for x in pair_shape]) * dsize/1e6 * 2

    mem_avail = max_memory - lib.current_memory()[0]
    mem_blk = 2*eris.nvir*eris.naux * dsize/1e6
    occ_blksize = max(1, min(nocc, int(np.floor((mem_avail-mem_est) * 0.7 / mem_blk))))
    log.debug1('occ_blksize for dOSV ERI: %d/%d', occ_blksize, nocc)

    J = tools.RaggedTensor2DTril(nocc, pair_shape, True, dtype) if with_J else None
    K = tools.RaggedTensor2DTril(nocc, pair_shape, True, dtype) if with_K else None

    for i in range(nocc):
        ivL = eris.get_occ_blk(i, i+1)[0]
        iAL = lib.dot(osv.u[i].T.conj(), ivL)
        for j in range(i+1):
            if not pair_mask[i, j]:
                continue
            if j == i:
                Kii = lib.dot(iAL, iAL.T.conj())
                if with_K:
                    K[i, j] = Kii
                if with_J:
                    J[i, j] = Kii
            else:
                jvL = eris.get_occ_blk(j, j+1)[0]
                if with_K:
                    jBL = lib.dot(osv.u[j].T.conj(), jvL)
                    K[i, j] = lib.dot(iAL, jBL.T.conj())
                    jBL = None
                if with_J:
                    jAL = lib.dot(osv.u[i].T.conj(), jvL)
                    iBL = lib.dot(osv.u[j].T.conj(), ivL)
                    J[i, j] = lib.dot(jAL, iBL.T.conj())
                    jAL = iBL = None
                jvL = None
        ivL = iAL = None

    return K, J


def get_dosv_eri_C(osv, eris, pair_mask, with_K=True, with_J=True, max_memory=None):
    ''' Calculate dOSV-type ERIs, i.e.,
            K[i,j](a_i, b_j) = (i a_i | j b_j)
            J[i,j](a_i, b_j) = (j a_i | i b_i)

        Args:
            with_K/J:
                If set to False, None will be returned for K/J

        Returns:
            K, J
    '''
    log = logger.new_logger(eris)

    if max_memory is None:
        max_memory = eris.max_memory

    dtype = np.float64
    dsize = 8

    nocc = osv.nocc
    nosv = osv.nosv
    pair_mask = tools._to_full(pair_mask, nocc)

    if not (with_J or with_K):
        return None, None

    pair_shape = [(nosv[i], nosv[j]) if pair_mask[i, j] else (0, 0)
                  for i in range(nocc) for j in range(i+1)]
    pair_size = [np.prod(x) for x in pair_shape]
    mem_est = sum([np.prod(x) for x in pair_shape]) * dsize/1e6 * 2

    mem_avail = max_memory - lib.current_memory()[0]
    mem_blk = 2*eris.nvir*eris.naux * dsize/1e6
    occ_blksize = max(1, min(nocc, int(np.floor((mem_avail-mem_est) * 0.7 / mem_blk))))
    log.debug1('occ_blksize for dOSV ERI: %d/%d', occ_blksize, nocc)

    if with_J:
        J = tools.RaggedTensor2DTril(nocc, pair_shape, True, dtype)
        Jptr = J.data.ctypes.data_as(ctypes.c_void_p)
    else:
        J = None
        Jptr = lib.c_null_ptr()
    if with_K:
        K = tools.RaggedTensor2DTril(nocc, pair_shape, True, dtype)
        Kptr = K.data.ctypes.data_as(ctypes.c_void_p)
    else:
        K = None
        Kptr = lib.c_null_ptr()

    if with_J and with_K:
        drv = libmp.get_dOSV_KJ
    elif with_J:
        drv = libmp.get_dOSV_J
    else:
        drv = libmp.get_dOSV_K

    for ibatch, (i0, i1) in enumerate(lib.prange(0, nocc, occ_blksize)):
        nocci = i1-i0
        iaL = eris.get_occ_blk(i0, i1)
        for jbatch, (j0, j1) in enumerate(lib.prange(0, nocc, occ_blksize)):
            noccj = j1-j0
            if jbatch > ibatch:
                continue
            if jbatch == ibatch:
                jbL = iaL
            else:
                jbL = eris.get_occ_blk(j0, j1)

            drv(
                Kptr, Jptr,
                np.asarray(pair_size, dtype=np.int32).ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(len(pair_size)),
                np.asarray(pair_mask[i0:i1, j0:j1],
                           dtype=np.int8, order='C').ctypes.data_as(ctypes.c_void_p),
                osv.u.data.ctypes.data_as(ctypes.c_void_p),
                np.asarray(osv.u.size, dtype=np.int32).ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(osv.u.ntensor),
                np.asarray(nosv, dtype=np.int32).ctypes.data_as(ctypes.c_void_p),
                iaL.ctypes.data_as(ctypes.c_void_p),
                jbL.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(i0), ctypes.c_int(j0),
                ctypes.c_int(nocci), ctypes.c_int(noccj),
                ctypes.c_int(eris.nvir), ctypes.c_int(eris.naux)
            )

    return K, J


def get_psv_eri(osv, eris, pair_mask, occ_energy, vir_energy, thresh_psv_lindep=1e-6,
                thresh_weakpair=0, pno_param=None, with_ex_ene=True, compress_diagpair=False,
                max_memory=None, pno_occ_topk=8):
    r''' Calculate PSV transformation matrix W in the OSV basis:
            a_ij = \sum_{b_i} W(b_i,a_ij) b_i + \sum_{b_j} W(b_j,a_ij) b_j
        and the PSV ERIs:
            K[i,j](a_ij, b_ij) = (i a_ij | j b_ij)

        The PSVs of pair (i,j) are obtained by combining OSVs of i and j and a subsequent
        orthogonalization and pseudocanonicalization. The PSVs can be further compressed
        using PNOs if `pno_param` is provided.

        Note that for diagonal pairs (i,i), the bare OSVs are used as PSVs without further
        treatment.

        Args:
            thresh_psv_lindep : float
                Threshold for linear dependency in orthogonalizing joint OSVs.
                Default: 1e-6.
            occ_energy : np.ndarray
                Diagonal elements of fock matrix in the LMO basis.
            pno_param : dict
                Parameters for PNO truncation:
                    thresh : float
                        Threshold for PNO occupation number.
                    thresh_ene : float
                        Threshold for PNO/PSV direct SC-MP2 pair energy ratio. If not
                        provided, energy-based selection is not used.

        Returns:
            e : RaggedTensor2DTril
                Pseudocanonicalized PSV energies.
            w : RaggedTensor2DTril
                OSV-to-PSV transformation matrix; non-zero only for pairs in `pair_mask`.
            K : RaggedTensor2DTril
                PSV ERIs; non-zero only for pairs in `pair_mask`.
            depair_pno : np.ndarray
                Pair energy correction due to PNO truncation.
    '''
    log = logger.new_logger(eris)

    with_pno = pno_param is not None
    pno_occ_topk = int(pno_occ_topk)

    nocc = osv.nocc
    nosv = osv.nosv

    pair_mask = tools._to_full(pair_mask, nocc)
    pair_mask_weak = np.zeros_like(pair_mask)

    es = np.ndarray((nocc, nocc), dtype=object)
    ws = np.ndarray((nocc, nocc), dtype=object)
    Ks = np.ndarray((nocc, nocc), dtype=object)

    tspans = np.zeros((8, 2))
    tnames = ['psv-cano', 'psv-gen', 'Kpsv', 'Epsv', 'pno', 'wpno', 'Kpno', 'dEpno']

    epair_psv_ss = np.zeros((nocc, nocc), dtype=np.float64)
    epair_psv_os = np.zeros((nocc, nocc), dtype=np.float64)
    epair_pno_ss = np.zeros((nocc, nocc), dtype=np.float64)
    epair_pno_os = np.zeros((nocc, nocc), dtype=np.float64)
    ntot = nocc*(nocc+1)//2
    pno_occ_top = np.zeros((ntot, pno_occ_topk), dtype=np.float64)
    for i in range(nocc):
        ivL = eris.get_occ_blk(i, i+1)[0]
        for j in range(i+1):
            if not pair_mask[i, j]:
                es[i, j] = np.zeros((0,))
                ws[i, j] = np.zeros((0, 0))
                Ks[i, j] = np.zeros((0, 0))
                continue
            if i == j and not compress_diagpair:
                es[i, i] = osv.e[i]
                ws[i, i] = np.eye(osv.u[i].shape[1])
                TICK = np.asarray((logger.process_clock(), logger.perf_counter()))
                iAL = lib.dot(osv.u[i].T.conj(), ivL)
                Ks[i, i] = lib.dot(iAL, iAL.T)
                TOCK = np.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[2] += TOCK - TICK
            else:
                fac = 2 if i != j else 1

                TICK = np.asarray((logger.process_clock(), logger.perf_counter()))
                s = np.eye(nosv[i]+nosv[j])
                s[:nosv[i], nosv[i]:] = osv.S[i, j]
                s[nosv[i]:, :nosv[i]] = osv.S[j, i]
                f = np.diag(np.hstack((osv.e[i], osv.e[j])))
                f[:nosv[i], nosv[i]:] = osv.F[i, j]
                f[nosv[i]:, :nosv[i]] = osv.F[j, i]
                epsv, wpsv = tools.safe_eigh(f, s, lindep_thr=thresh_psv_lindep)
                TOCK = np.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[0] += TOCK - TICK

                upsv = lib.dot(osv.u[i], wpsv[:nosv[i]])
                upsv += lib.dot(osv.u[j], wpsv[nosv[i]:])
                TICK = np.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[1] += TICK - TOCK

                iAL = lib.dot(upsv.T.conj(), ivL)
                jvL = eris.get_occ_blk(j, j+1)[0]
                jBL = lib.dot(upsv.T.conj(), jvL)
                jvL = None
                Kpsv = lib.dot(iAL, jBL.T)
                TOCK = np.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[2] += TOCK - TICK

                t2psv = Kpsv.conj() / (occ_energy[i]+occ_energy[j] - (epsv[:, None]+epsv))
                ed = einsum('ab,ab->', t2psv, Kpsv).real
                if with_ex_ene:
                    ex = -einsum('ab,ba->', t2psv, Kpsv).real
                else:
                    ex = 0
                eij_psv_ss = (ed + ex) * fac
                eij_psv_os = (ed) * fac
                eij_psv = eij_psv_ss + eij_psv_os

                TICK = np.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[3] += TICK - TOCK

                if abs(eij_psv) >= thresh_weakpair:
                    if with_pno:
                        epno, upno, occ_act, occ_all = get_pno1(
                            Kpsv, t2psv, epsv, return_occ=True, **pno_param)
                        idx = i*(i+1)//2 + j
                        ncopy = min(pno_occ_topk, occ_act.size)
                        if ncopy:
                            pno_occ_top[idx, :ncopy] = occ_act[:ncopy]

                        TOCK = np.asarray((logger.process_clock(), logger.perf_counter()))
                        tspans[4] += TOCK - TICK

                        wpno = lib.dot(wpsv, upno)

                        TICK = np.asarray((logger.process_clock(), logger.perf_counter()))
                        tspans[5] += TICK - TOCK

                        Kpno = reduce(lib.dot, (upno.T.conj(), Kpsv, upno))

                        TOCK = np.asarray((logger.process_clock(), logger.perf_counter()))
                        tspans[6] += TOCK - TICK

                        Tpno = Kpno.conj() / (occ_energy[i]+occ_energy[j] - (epno[:, None] + epno))

                        ed = einsum('ab,ab->', Tpno, Kpno).real
                        if with_ex_ene:
                            ex = -einsum('ab,ba->', Tpno, Kpno).real
                        else:
                            ex = 0
                        eij_pno_ss = (ed + ex) * fac
                        eij_pno_os = (ed) * fac
                        epair_pno_ss[i, j] = eij_psv_ss - eij_pno_ss
                        epair_pno_os[i, j] = eij_psv_os - eij_pno_os

                        TICK = np.asarray((logger.process_clock(), logger.perf_counter()))
                        tspans[7] += TICK - TOCK

                        es[i, j] = epno
                        ws[i, j] = wpno
                        Ks[i, j] = Kpno
                    else:
                        es[i, j] = epsv
                        ws[i, j] = wpsv
                        Ks[i, j] = Kpsv
                else:  # weak-pair
                    pair_mask_weak[i, j] = pair_mask_weak[j, i] = True
                    epair_psv_ss[i, j] = eij_psv_ss
                    epair_psv_os[i, j] = eij_psv_os
                    es[i, j] = np.zeros((0,))
                    ws[i, j] = np.zeros((0, 0))
                    Ks[i, j] = np.zeros((0, 0))
        ivL = None

    epair_psv_ss = lib.pack_tril(epair_psv_ss)
    epair_psv_os = lib.pack_tril(epair_psv_os)
    epair_psv = lib.tag_array(epair_psv_ss+epair_psv_os,
                              e_corr_ss=epair_psv_ss, e_corr_os=epair_psv_os)

    epair_pno_ss = lib.pack_tril(epair_pno_ss)
    epair_pno_os = lib.pack_tril(epair_pno_os)
    epair_pno = lib.tag_array(epair_pno_ss+epair_pno_os,
                              e_corr_ss=epair_pno_ss, e_corr_os=epair_pno_os)

    shape = [es[i, j].shape for i in range(nocc) for j in range(i+1)]
    e = tools.RaggedTensor2DTril(nocc, shape, False)
    shape = [ws[i, j].shape for i in range(nocc) for j in range(i+1)]
    w = tools.RaggedTensor2DTril(nocc, shape, False)
    shape = [Ks[i, j].shape for i in range(nocc) for j in range(i+1)]
    K = tools.RaggedTensor2DTril(nocc, shape, True)
    for i in range(nocc):
        for j in range(i+1):
            if pair_mask[i, j]:
                w[i, j] = ws[i, j]
                e[i, j] = es[i, j]
                K[i, j] = Ks[i, j]

    if not with_pno:
        tspans = tspans[:2]
        tnames = tnames[:2]
    for tspan, tname in zip(tspans, tnames):
        log.debug('make_psv CPU time for %-10s  %9.2f sec  wall time %9.2f sec', tname, *tspan)
    log.info('')

    return pair_mask_weak, e, w, K, epair_psv, epair_pno, pno_occ_top


def get_psv_eri_C(osv, eris, pair_mask, occ_energy, vir_energy, thresh_psv_lindep=1e-6,
                  thresh_weakpair=0, pno_param=None, with_ex_ene=True, compress_diagpair=False,
                  max_memory=None):
    r''' Calculate PSV transformation matrix W in the OSV basis:
            a_ij = \sum_{b_i} W(b_i,a_ij) b_i + \sum_{b_j} W(b_j,a_ij) b_j
        and the PSV ERIs:
            K[i,j](a_ij, b_ij) = (i a_ij | j b_ij)

        The PSVs of pair (i,j) are obtained by combining OSVs of i and j and a subsequent
        orthogonalization and pseudocanonicalization. The PSVs can be further compressed
        using PNOs if `pno_param` is provided.

        Note that for diagonal pairs (i,i), the bare OSVs are used as PSVs without further
        treatment.

        Args:
            thresh_psv_lindep : float
                Threshold for linear dependency in orthogonalizing joint OSVs.
                Default: 1e-6.
            occ_energy : np.ndarray
                Diagonal elements of fock matrix in the LMO basis.
            pno_param : dict
                Parameters for PNO truncation:
                    thresh : float
                        Threshold for PNO occupation number.
                    thresh_ene : float
                        Threshold for PNO/PSV direct SC-MP2 pair energy ratio. If not
                        provided, energy-based selection is not used.

        Returns:
            e : RaggedTensor2DTril
                Pseudocanonicalized PSV energies.
            w : RaggedTensor2DTril
                OSV-to-PSV transformation matrix; non-zero only for pairs in `pair_mask`.
            K : RaggedTensor2DTril
                PSV ERIs; non-zero only for pairs in `pair_mask`.
            depair_pno : np.ndarray
                Pair energy correction due to PNO truncation.
    '''
    assert(not compress_diagpair)

    from pyscf.ao2mo.outcore import balance_partition
    log = logger.new_logger(eris)

    with_pno = pno_param is not None
    if max_memory is None:
        max_memory = eris.max_memory

    nocc = osv.nocc
    nosv = osv.nosv
    nvir = vir_energy.size

    dsize = 8

    pair_mask = tools._to_full(pair_mask, nocc)
    pair_mask_weak = np.zeros_like(pair_mask)

    tspans = np.zeros((7, 2))
    tnames = ['psv', 'Kpsv', 'Epsv', 'pno', 'wpno', 'Kpno', 'dEpno']

    es = np.ndarray((nocc, nocc), dtype=object)
    ws = np.ndarray((nocc, nocc), dtype=object)
    Ks = np.ndarray((nocc, nocc), dtype=object)

    epair_psv_ss = np.zeros((nocc, nocc), dtype=np.float64)
    epair_psv_os = np.zeros((nocc, nocc), dtype=np.float64)
    epair_pno_ss = np.zeros((nocc, nocc), dtype=np.float64)
    epair_pno_os = np.zeros((nocc, nocc), dtype=np.float64)

    # diagonal pairs and dist pairs
    for i in range(nocc):
        ivL = eris.get_occ_blk(i, i+1)[0]
        for j in range(i+1):
            if not pair_mask[i, j]:
                es[i, j] = np.zeros((0,))
                ws[i, j] = np.zeros((0, 0))
                Ks[i, j] = np.zeros((0, 0))
                continue
            if i == j and not compress_diagpair:
                es[i, i] = osv.e[i]
                ws[i, i] = np.eye(osv.u[i].shape[1])
                TICK = np.asarray((logger.process_clock(), logger.perf_counter()))
                iAL = lib.dot(osv.u[i].T.conj(), ivL)
                Ks[i, i] = lib.dot(iAL, iAL.T)
                TOCK = np.asarray((logger.process_clock(), logger.perf_counter()))
                tspans[1] += TOCK - TICK

    ASINTARRAY = lambda x: np.asarray(x, order='C', dtype=np.int32)
    pairs = ASINTARRAY(np.hstack([(i, j) for i in range(nocc) for j in
                                  np.where(pair_mask[i, :i])[0]]))
    npsv_by_pair = np.asarray([nosv[i]+nosv[j] for i, j in zip(pairs[::2], pairs[1::2])])
    size_loc = np.cumsum(np.concatenate(([0], npsv_by_pair**2))).astype(int)
    mem_avail = max_memory - lib.current_memory()[0]
    # 2 copies of Kpsv & wpsv (1 copy is buf) --> 4 copies of size-npsv^2 arrays in total
    buflen = max(1, min(size_loc[-1], int(np.floor(mem_avail*0.8 / 4 / (dsize/1e6)))))
    pair_range = balance_partition(size_loc, buflen)
    buflen = max([x[2] for x in pair_range])
    drv_psv = libmp.get_psv
    drv_psv_eri = libmp.get_psv_eri

    for start, end, _ in pair_range:
        TICK = np.asarray((logger.process_clock(), logger.perf_counter()))

        npair1 = end - start
        pairs1 = pairs[2*start:2*end]
        shape_epsv = [(n,) for n in npsv_by_pair[start:end]]
        shape_wpsv = [(n, n) for n in npsv_by_pair[start:end]]
        epsvs1 = tools.RaggedTensor1D(shape_epsv)
        wpsvs1 = tools.RaggedTensor1D(shape_wpsv)
        npsvs1 = np.zeros(npair1, dtype=np.int32)

        drv_psv(
            epsvs1.data.ctypes.data_as(ctypes.c_void_p),
            wpsvs1.data.ctypes.data_as(ctypes.c_void_p),
            npsvs1.ctypes.data_as(ctypes.c_void_p),
            ASINTARRAY(epsvs1.size).ctypes.data_as(ctypes.c_void_p),
            ASINTARRAY(wpsvs1.size).ctypes.data_as(ctypes.c_void_p),

            pairs1.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(npair1),

            osv.e.data.ctypes.data_as(ctypes.c_void_p),
            ASINTARRAY(osv.e.size).ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(osv.e.ntensor),

            osv.F.data.ctypes.data_as(ctypes.c_void_p),
            osv.S.data.ctypes.data_as(ctypes.c_void_p),
            ASINTARRAY(osv.F.size).ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(osv.F.ntensor),

            np.asarray(nosv, order='C', dtype=np.int32).ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nocc),
            ctypes.c_int(nvir),

            ctypes.c_double(thresh_psv_lindep),
        )

        TOCK = np.asarray((logger.process_clock(), logger.perf_counter()))
        tspans[0] += TOCK - TICK

        Kpsvs1 = tools.RaggedTensor1D([(n, n) for n in npsvs1])
        drv_psv_eri(
            Kpsvs1.data.ctypes.data_as(ctypes.c_void_p),
            epsvs1.data.ctypes.data_as(ctypes.c_void_p),
            wpsvs1.data.ctypes.data_as(ctypes.c_void_p),
            npsvs1.ctypes.data_as(ctypes.c_void_p),
            ASINTARRAY(Kpsvs1.size).ctypes.data_as(ctypes.c_void_p),
            ASINTARRAY(epsvs1.size).ctypes.data_as(ctypes.c_void_p),
            ASINTARRAY(wpsvs1.size).ctypes.data_as(ctypes.c_void_p),

            pairs1.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(npair1),

            osv.u.data.ctypes.data_as(ctypes.c_void_p),
            ASINTARRAY(osv.u.size).ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(osv.e.ntensor),

            np.asarray(nosv, order='C', dtype=np.int32).ctypes.data_as(ctypes.c_void_p),

            eris.ovL.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nocc),
            ctypes.c_int(nvir),
            ctypes.c_int(eris.ovL.shape[-1]),
        )

        TICK = np.asarray((logger.process_clock(), logger.perf_counter()))
        tspans[1] += TICK - TOCK

        for ipair, (i, j) in enumerate(zip(pairs1[::2], pairs1[1::2])):
            fac = 2 if i != j else 1

            nij = npsvs1[ipair]
            epsv = epsvs1[ipair][:nij]
            wpsv = wpsvs1[ipair][:nij].T
            Kpsv = Kpsvs1[ipair]

            TOCK = np.asarray((logger.process_clock(), logger.perf_counter()))

            t2psv = Kpsv.conj() / (occ_energy[i]+occ_energy[j] - (epsv[:, None]+epsv))
            ed = einsum('ab,ab->', t2psv, Kpsv).real
            if with_ex_ene:
                ex = -einsum('ab,ba->', t2psv, Kpsv).real
            else:
                ex = 0
            eij_psv_ss = (ed + ex) * fac
            eij_psv_os = (ed) * fac
            eij_psv = eij_psv_ss + eij_psv_os

            TICK = np.asarray((logger.process_clock(), logger.perf_counter()))
            tspans[2] += TICK - TOCK

            if abs(eij_psv) >= thresh_weakpair:
                if with_pno:
                    epno, upno = get_pno1(Kpsv, t2psv, epsv, **pno_param)

                    TOCK = np.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[3] += TOCK - TICK

                    wpno = lib.dot(wpsv, upno)

                    TICK = np.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[4] += TICK - TOCK

                    Kpno = reduce(lib.dot, (upno.T.conj(), Kpsv, upno))

                    TOCK = np.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[5] += TOCK - TICK

                    Tpno = Kpno.conj() / (occ_energy[i]+occ_energy[j] - (epno[:, None] + epno))

                    ed = einsum('ab,ab->', Tpno, Kpno).real
                    if with_ex_ene:
                        ex = -einsum('ab,ba->', Tpno, Kpno).real
                    else:
                        ex = 0
                    eij_pno_ss = (ed + ex) * fac
                    eij_pno_os = (ed) * fac
                    epair_pno_ss[i, j] = eij_psv_ss - eij_pno_ss
                    epair_pno_os[i, j] = eij_psv_os - eij_pno_os

                    TICK = np.asarray((logger.process_clock(), logger.perf_counter()))
                    tspans[6] += TICK - TOCK

                    es[i, j] = epno
                    ws[i, j] = wpno
                    Ks[i, j] = Kpno
                else:
                    es[i, j] = epsv
                    ws[i, j] = wpsv
                    Ks[i, j] = Kpsv
            else:  # weak-pair
                pair_mask_weak[i, j] = pair_mask_weak[j, i] = True
                epair_psv_ss[i, j] = eij_psv_ss
                epair_psv_os[i, j] = eij_psv_os
                es[i, j] = np.zeros((0,))
                ws[i, j] = np.zeros((0, 0))
                Ks[i, j] = np.zeros((0, 0))

    epair_psv_ss = lib.pack_tril(epair_psv_ss)
    epair_psv_os = lib.pack_tril(epair_psv_os)
    epair_psv = lib.tag_array(epair_psv_ss+epair_psv_os,
                              e_corr_ss=epair_psv_ss, e_corr_os=epair_psv_os)

    epair_pno_ss = lib.pack_tril(epair_pno_ss)
    epair_pno_os = lib.pack_tril(epair_pno_os)
    epair_pno = lib.tag_array(epair_pno_ss+epair_pno_os,
                              e_corr_ss=epair_pno_ss, e_corr_os=epair_pno_os)

    shape = [es[i, j].shape for i in range(nocc) for j in range(i+1)]
    e = tools.RaggedTensor2DTril(nocc, shape, False)
    shape = [ws[i, j].shape for i in range(nocc) for j in range(i+1)]
    w = tools.RaggedTensor2DTril(nocc, shape, False)
    shape = [Ks[i, j].shape for i in range(nocc) for j in range(i+1)]
    K = tools.RaggedTensor2DTril(nocc, shape, True)
    for i in range(nocc):
        for j in range(i+1):
            if pair_mask[i, j]:
                w[i, j] = ws[i, j]
                e[i, j] = es[i, j]
                K[i, j] = Ks[i, j]

    if not with_pno:
        tspans = tspans[:2]
        tnames = tnames[:2]
    for tspan, tname in zip(tspans, tnames):
        log.debug('make_psv CPU time for %-10s  %9.2f sec  wall time %9.2f sec', tname, *tspan)
    log.info('')

    return pair_mask_weak, e, w, K, epair_psv, epair_pno


class PairDomainBase(lib.StreamObject):
    '''
        Attributes:
            nocc (int):
                Number of occupied orbitals.
            pair (1d array):
                List of pairs (i,j) for i >= j.
            npair (int):
                Number of pairs.
            pair_mask (2d array):
                2d bool array of shape (nocc,nocc).

        Methods:
            loop_pair : no args
                Return list of connected (i,j) pairs.
            loop_j : (i, tril=True)
                Return list of j indices that are connected to i.
            loop_k : (i, j)
                Return list of k indices that are connected to i and j
    '''
    def __init__(self, nocc, pair_mask=None):
        self.nocc = nocc
        self._pair = None
        self._pair_mask = None
        self._pair_mask_full = None

        if pair_mask is None:
            pair_mask = np.ones((nocc, nocc), dtype=bool)

        self.update_pair_(pair_mask)

    @property
    def pair(self):
        if self._pair is None:
            self.update_pair_()
        return self._pair

    @property
    def pair_mask(self):
        return self._pair_mask

    @property
    def npair(self):
        return len(self.pair)

    def update_pair_(self, pair_mask):
        nocc = self.nocc
        if pair_mask.size == nocc*(nocc+1)//2:
            self._pair_mask = lib.unpack_tril(pair_mask)
        elif pair_mask.size == nocc*nocc:
            self._pair_mask = pair_mask
        else:
            raise ValueError('Input pair mask has wrong shape: must be nocc^2 or nocc*(nocc+1)/2.')
        self._pair = [(i, j) for i in range(nocc) for j in self.loop_j(i)]

    def loop_pair(self):
        return self.pair

    def loop_j(self, i, tril=True):
        jmax = i+1 if tril else self.nocc
        return np.where(self.pair_mask[i, :jmax])[0]

    def loop_k(self, i, j):
        return np.where(self.pair_mask[i] & self.pair_mask[j])[0]


class PairDomain_dOSV(PairDomainBase):
    ''' dOSV-based pair domain. Purely python-based slow implementation.

        Additional attributes:
            osv (OSV object):
                Provides osv.u, osv.e, osv.nosv.
            S (RaggedTensor2DTril object):
                OSV overlap. Calculated by `osv.get_osv_ovlp()`.
            K (RaggedTensor2DTril):
                K[i,j] = (i a_i | j b_j) is a matrix of size nosv[i] * nosv[j].
            J (RaggedTensor2DTril):
                J[i,j] = (j a_i | i b_j) is a matrix of size nosv[i] * nosv[j].

        Method:
            new_array:
                No input. Returns an empty RaggedTensor2DTril object of same shape as K/J.
    '''
    def __init__(self, eris, osv, vir_energy, pair_mask=None, with_J=True):
        PairDomainBase.__init__(self, osv.nocc, pair_mask)
        self.osv = osv
        self.pair_shape = [(osv.nosv[i], osv.nosv[j]) if self.pair_mask[i, j] else (0, 0)
                           for i in range(self.nocc) for j in range(i+1)]

        self.K = self.J = None

        self.dsize = 8
        self.dtype = np.float64

        self.build(eris, with_J)

    @property
    def S(self):
        return self.osv.S

    def build(self, eris, with_J):
        self.K, self.J = self._build_KJ(eris, with_J)

    def _build_KJ(self, eris, with_J):
        return get_dosv_eri(self.osv, eris, self.pair_mask, with_J=with_J)

    def _build_J(self, eris):
        self.J = get_dosv_eri(self.osv, eris, self.pair_mask, with_K=False, with_J=True)[1]

    def mem_array(self):
        return np.sum([np.prod(x) for x in self.pair_shape]) * self.dsize/1e6

    def new_array(self, transpose=True):
        return tools.RaggedTensor2DTril(self.nocc, self.pair_shape, transpose, self.dtype)

    def update_(self, pair_mask):
        ''' Update pair info (pair/pair_mask/pair_shape) and integrals (K/J). The truth
            set of the input pair_mask must be a subset of the old pair_mask.
        '''
        self.update_pair_(pair_mask)

        nosv = self.osv.nosv
        self.pair_shape = [(nosv[i], nosv[j]) if self.pair_mask[i, j] else (0, 0)
                           for i in range(self.nocc) for j in range(i+1)]

        def update1(X):
            if X is None:
                return X
            X1 = self.new_array()
            for i, j in self.loop_pair():
                X1[i, j] = X[i, j]
            return X1

        self.K = update1(self.K)
        self.J = update1(self.J)


class PairDomain_dOSV_C(PairDomain_dOSV):
    def _build_KJ(self, eris, with_J):
        return get_dosv_eri_C(self.osv, eris, self.pair_mask, with_J=with_J)


class PairDomain_OSV(PairDomain_dOSV):
    '''
        Attributes:
            osv : OSV object
                osv.u, osv.e, osv.nosv.
            npsv : 1D list of length nocc
                Number of OSVs for each occupied.
            w : RaggedTensor2D object
                Joint OSV-to-PSV transforma matrix.
            e : RaggedTensor2D object
                PSV energy (pseudocanonicalized).
            K : RaggedTensor2D object
                K[i,j](a_ij, b_ij) := (i a_ij | j b_ij)

        Methods:
            new_array:
                No input. Returns an empty RaggedTensor2DTril object of same shape as K.
    '''
    def __init__(self, eris, osv, occ_energy, vir_energy, pair_mask=None, thresh_psv_lindep=1e-6,
                 thresh_weakpair=0):
        PairDomainBase.__init__(self, osv.nocc, pair_mask=pair_mask)
        self.osv = osv

        self.w = None
        self.e = None
        self.npsv = None
        self.npsv_tril = None
        self.K = None

        self.dsize = 8
        self.dtype = np.float64

        self.build(eris, occ_energy, vir_energy, thresh_psv_lindep, thresh_weakpair)

    def build(self, eris, occ_energy, vir_energy, thresh_psv_lindep, thresh_weakpair):
        nocc = self.nocc
        pair_mask_weak, self.e, self.w, self.K, epair_psv = \
            get_psv_eri(self.osv, eris, self.pair_mask,
                        occ_energy, vir_energy,
                        thresh_psv_lindep=thresh_psv_lindep,
                        thresh_weakpair=thresh_weakpair)
        self.update_pair_(self.pair_mask & ~pair_mask_weak)
        self.epair_psv_ss = epair_psv.e_corr_ss
        self.epair_psv_os = epair_psv.e_corr_os
        self.epair_psv = np.asarray(epair_psv)

        self.npsv = np.zeros((nocc, nocc), dtype=np.int32)
        for i in range(nocc):
            for j in range(i+1):
                if self.pair_mask[i, j]:
                    self.npsv[i, j] = self.npsv[j, i] = self.e[i, j].size
        self.npsv_tril = lib.pack_tril(self.npsv)
        self.pair_shape = [(self.npsv[i, j],)*2 if self.pair_mask[i, j] else (0, 0)
                           for i in range(nocc) for j in range(i+1)]

    def get_u(self, i, j):
        u = self.osv.u
        if i == j:
            return u[i]
        else:
            k, l = (i, j) if i >= j else (j, i)
            w = self.w[k, l]
            nk = self.osv.nosv[k]
            return lib.dot(u[k], w[:nk]) + lib.dot(u[l], w[nk:])

    def get_ovlp(self, i, j, k, l):
        osv = self.osv
        psv = self
        if i < j:
            i, j = j, i
        if k < l:
            k, l = l, k
        if i == j and k == l:
            return self.S[i, k]
        elif i == j:
            w = psv.w[k, l]
            nk = osv.nosv[k]
            outS = w[:nk].copy() if i == k else lib.dot(self.S[i, k], w[:nk])
            outS += w[nk:] if i == l else lib.dot(self.S[i, l], w[nk:])
            return outS
        elif k == l:
            w = psv.w[i, j]
            ni = osv.nosv[i]
            outS = w[:ni].T.conj().copy() if i == k else lib.dot(w[:ni].T.conj(), self.S[i, k])
            outS += w[ni:].T.conj() if j == k else lib.dot(w[ni:].T.conj(), self.S[j, k])
            return outS
        else:
            wij = psv.w[i, j]
            wkl = psv.w[k, l]
            ni = osv.nosv[i]
            nk = osv.nosv[k]
            outS = (lib.dot(wij[:ni].T.conj(), wkl[:nk]) if i == k else
                    reduce(lib.dot, (wij[:ni].T.conj(), self.S[i, k], wkl[:nk])))
            outS += (lib.dot(wij[:ni].T.conj(), wkl[nk:]) if i == l else
                     reduce(lib.dot, (wij[:ni].T.conj(), self.S[i, l], wkl[nk:])))
            outS += (lib.dot(wij[ni:].T.conj(), wkl[:nk]) if j == k else
                     reduce(lib.dot, (wij[ni:].T.conj(), self.S[j, k], wkl[:nk])))
            outS += (lib.dot(wij[ni:].T.conj(), wkl[nk:]) if j == l else
                     reduce(lib.dot, (wij[ni:].T.conj(), self.S[j, l], wkl[nk:])))
            return outS


class PairDomain_PNO(PairDomain_OSV):
    def __init__(self, eris, osv, pno_param, occ_energy, vir_energy, pair_mask=None,
                 thresh_psv_lindep=1e-6, thresh_weakpair=0, with_ex_ene=True,
                 compress_diagpair=False, pno_occ_topk=8):
        PairDomainBase.__init__(self, osv.nocc, pair_mask=pair_mask)
        self.osv = osv

        self.w = None
        self.e = None
        self.npsv = None
        self.npsv_tril = None
        self.K = None
        self.pno_occ_top = None

        self.dsize = 8
        self.dtype = np.float64

        self.build(eris, pno_param, occ_energy, vir_energy, thresh_psv_lindep, thresh_weakpair,
                   with_ex_ene, compress_diagpair, pno_occ_topk)

    def build(self, eris, pno_param, occ_energy, vir_energy, thresh_psv_lindep, thresh_weakpair,
              with_ex_ene, compress_diagpair, pno_occ_topk):
        nocc = self.nocc
        pair_mask_weak, self.e, self.w, self.K, epair_psv, epair_pno, self.pno_occ_top = \
            get_psv_eri(self.osv, eris, self.pair_mask,
                        occ_energy, vir_energy,
                        thresh_psv_lindep=thresh_psv_lindep,
                        thresh_weakpair=thresh_weakpair,
                        pno_param=pno_param,
                        with_ex_ene=with_ex_ene,
                        compress_diagpair=compress_diagpair,
                        pno_occ_topk=pno_occ_topk)
        self.update_pair_(self.pair_mask & ~pair_mask_weak)
        self.epair_psv_ss = epair_psv.e_corr_ss
        self.epair_psv_os = epair_psv.e_corr_os
        self.epair_psv = np.asarray(epair_psv)
        if pno_param is not None:
            self.epair_pno_ss = epair_pno.e_corr_ss
            self.epair_pno_os = epair_pno.e_corr_os
            self.epair_pno = np.asarray(epair_pno)

        self.npsv = np.zeros((nocc, nocc), dtype=np.int32)
        for i in range(nocc):
            for j in range(i+1):
                if self.pair_mask[i, j]:
                    self.npsv[i, j] = self.npsv[j, i] = self.e[i, j].size
        self.npsv_tril = lib.pack_tril(self.npsv)
        self.pair_shape = [(self.npsv[i, j],)*2 if self.pair_mask[i, j] else (0, 0)
                           for i in range(nocc) for j in range(i+1)]


class PairDomain_PNO_C(PairDomain_PNO):
    def build(self, eris, pno_param, occ_energy, vir_energy, thresh_psv_lindep, thresh_weakpair,
              with_ex_ene, compress_diagpair, pno_occ_topk):
        nocc = self.nocc
        pair_mask_weak, self.e, self.w, self.K, epair_psv, epair_pno = \
            get_psv_eri_C(self.osv, eris, self.pair_mask,
                          occ_energy, vir_energy,
                          thresh_psv_lindep=thresh_psv_lindep,
                          thresh_weakpair=thresh_weakpair,
                          pno_param=pno_param,
                          with_ex_ene=with_ex_ene,
                          compress_diagpair=compress_diagpair)
        self.update_pair_(self.pair_mask & ~pair_mask_weak)
        self.epair_psv_ss = epair_psv.e_corr_ss
        self.epair_psv_os = epair_psv.e_corr_os
        self.epair_psv = np.asarray(epair_psv)
        if pno_param is not None:
            self.epair_pno_ss = epair_pno.e_corr_ss
            self.epair_pno_os = epair_pno.e_corr_os
            self.epair_pno = np.asarray(epair_pno)
        self.pno_occ_top = None

        self.npsv = np.zeros((nocc, nocc), dtype=np.int32)
        for i in range(nocc):
            for j in range(i+1):
                if self.pair_mask[i, j]:
                    self.npsv[i, j] = self.npsv[j, i] = self.e[i, j].size
        self.npsv_tril = lib.pack_tril(self.npsv)
        self.pair_shape = [(self.npsv[i, j],)*2 if self.pair_mask[i, j] else (0, 0)
                           for i in range(nocc) for j in range(i+1)]
