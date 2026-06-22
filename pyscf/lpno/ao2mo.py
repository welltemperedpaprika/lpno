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

"""Build DF ERIs in the occupied-virtual block."""

import ctypes
import numpy as np
import scipy.linalg
import h5py
from pyscf.df import addons
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo

try:
    libmp = lib.load_library('libmp')
except Exception:
    libmp = None


THRESH_LINDEP = 1e-10


def get_eris(mylpno, vir_coeff=None, ovL=None, ovL_to_save=None):
    """Build or load the DF ERI helper used by LPnO methods."""
    log = logger.new_logger(mylpno)

    with_df = getattr(mylpno, 'with_df', None)
    assert with_df is not None
    if with_df.auxmol is None:
        with_df.auxmol = addons.make_auxmol(with_df.mol, with_df.auxbasis)
    naux = with_df.auxmol.nao_nr()
    if with_df._cderi is None:
        log.debug('Caching ovL-type integrals directly')

    else:
        log.debug('Caching ovL-type integrals by transforming saved AO 3c integrals.')

    lo_coeff = mylpno.lo_coeff
    if vir_coeff is None:
        vir_coeff = mylpno.split_mo_coeff()[2]

    nocc = lo_coeff.shape[1]
    nvir = vir_coeff.shape[1]

    if ovL is not None:
        if isinstance(ovL, np.ndarray):
            outcore = False
        elif isinstance(ovL, str):
            outcore = True
        else:
            log.error(
                'Unknown data type %s for input `ovL` (should be np.ndarray or str).',
                type(ovL),
            )
            raise TypeError
    else:
        mem_now = mylpno.max_memory - lib.current_memory()[0]
        mem_df = nocc * nvir * naux * 8 / 1024**2.0
        log.debug('ao2mo est mem= %.2f MB  avail mem= %.2f MB', mem_df, mem_now)
        if mylpno.force_outcore:
            outcore = True
        else:
            outcore = (ovL_to_save is not None) or (mem_now * 0.8 < mem_df)
    log.debug('ovL-type integrals are cached %s', 'outcore' if outcore else 'incore')

    if outcore:
        eris = _DFOUTCOREERIS(
            with_df,
            lo_coeff,
            vir_coeff,
            mylpno.max_memory,
            ovL=ovL,
            ovL_to_save=ovL_to_save,
            verbose=log.verbose,
            stdout=log.stdout,
        )
    else:
        eris = _DFINCOREERIS(
            with_df,
            lo_coeff,
            vir_coeff,
            mylpno.max_memory,
            ovL=ovL,
            verbose=log.verbose,
            stdout=log.stdout,
        )
    eris.build()

    return eris


class _DFINCOREERIS:
    def __init__(
        self,
        with_df,
        occ_coeff,
        vir_coeff,
        max_memory,
        ovL=None,
        verbose=None,
        stdout=None,
    ):
        self.with_df = with_df
        self.occ_coeff = occ_coeff
        self.vir_coeff = vir_coeff

        self.max_memory = max_memory
        self.verbose = verbose
        self.stdout = stdout

        self.dtype = self.occ_coeff.dtype
        assert self.dtype == np.float64  # FIXME: support complex
        self.dsize = 8

        self.ovL = ovL

    @property
    def nocc(self):
        return self.occ_coeff.shape[1]

    @property
    def nvir(self):
        return self.vir_coeff.shape[1]

    @property
    def naux(self):
        return self.ovL.shape[-1]

    def build(self):
        log = logger.new_logger(self)
        is_pbc = hasattr(self.with_df, 'cell')
        if self.ovL is None:
            if self.with_df._cderi is None and not is_pbc:
                self.ovL = _init_mp_df_eris_direct(
                    self.with_df,
                    self.occ_coeff,
                    self.vir_coeff,
                    self.max_memory,
                    log=log,
                )
            else:
                self.ovL = _init_mp_df_eris(
                    self.with_df,
                    self.occ_coeff,
                    self.vir_coeff,
                    self.max_memory,
                    log=log,
                )

    def get_occ_blk(self, i0, i1):
        return np.asarray(self.ovL[i0:i1], order='C')


class _DFOUTCOREERIS(_DFINCOREERIS):
    def __init__(
        self,
        with_df,
        occ_coeff,
        vir_coeff,
        max_memory,
        ovL=None,
        ovL_to_save=None,
        verbose=None,
        stdout=None,
    ):
        _DFINCOREERIS.__init__(
            self, with_df, occ_coeff, vir_coeff, max_memory, None, verbose, stdout
        )

        self._ovL = ovL
        self._ovL_to_save = ovL_to_save

    def build(self):
        log = logger.new_logger(self)
        with_df = self.with_df
        if self._ovL is None:
            if isinstance(self._ovL_to_save, str):
                self.feri = h5py.File(self._ovL_to_save, 'w')
            else:
                self.feri = lib.H5TmpFile()
            log.info('ovL is saved to %s', self.feri.filename)
            is_pbc = hasattr(with_df, 'cell')
            if with_df._cderi is None and not is_pbc:
                _init_mp_df_eris_direct(
                    with_df,
                    self.occ_coeff,
                    self.vir_coeff,
                    self.max_memory,
                    h5obj=self.feri,
                    log=log,
                )
            else:
                _init_mp_df_eris(
                    with_df,
                    self.occ_coeff,
                    self.vir_coeff,
                    self.max_memory,
                    h5obj=self.feri,
                    log=log,
                )
            self.ovL = self.feri['ovL']
        elif isinstance(self._ovL, str):
            self.feri = h5py.File(self._ovL, 'r')
            log.info('ovL is read from %s', self.feri.filename)
            assert 'ovL' in self.feri
            self.ovL = self.feri['ovL']
        else:
            raise RuntimeError


def _init_mp_df_eris(with_df, occ_coeff, vir_coeff, max_memory, h5obj=None, log=None):
    if log is None:
        log = logger.new_logger(with_df)

    nao, nocc = occ_coeff.shape
    nvir = vir_coeff.shape[1]
    nmo = nocc + nvir
    nao_pair = nao**2
    naux = with_df.get_naoaux()

    dtype = occ_coeff.dtype
    assert dtype == np.float64
    dsize = 8

    mo = np.asarray(np.hstack((occ_coeff, vir_coeff)), order='F')
    ijslice = (0, nocc, nocc, nmo)

    if h5obj is None:  # incore
        ovL = np.empty((nocc, nvir, naux), dtype=dtype)
    else:
        ovL_shape = (nocc, nvir, naux)
        ovL = h5obj.create_dataset(
            'ovL', ovL_shape, dtype=dtype, chunks=(1, *ovL_shape[1:])
        )

    mem_avail = max_memory - lib.current_memory()[0]

    if isinstance(ovL, np.ndarray):
        # incore: batching aux (OV + Nao_pair) * [X] = M
        mem_auxblk = (nao_pair + nocc * nvir) * dsize / 1e6
        aux_blksize = min(naux, max(1, int(np.floor(mem_avail * 0.5 / mem_auxblk))))
        log.debug('aux blksize for incore ao2mo: %d/%d', aux_blksize, naux)
        buf = np.empty(aux_blksize * nocc * nvir, dtype=dtype)
        ijslice = (0, nocc, nocc, nmo)

        p1 = 0
        for Lpq in with_df.loop(blksize=aux_blksize):
            p0, p1 = p1, p1 + Lpq.shape[0]
            out = _ao2mo.nr_e2(Lpq, mo, ijslice, aosym='s2', out=buf)
            ovL[:, :, p0:p1] = out.reshape(-1, nocc, nvir).transpose(1, 2, 0)
            Lpq = out = None
        buf = None
    else:
        # outcore: batching occ [O]XV and aux ([O]V + Nao_pair)*[X]
        mem_occblk = naux * nvir * dsize / 1e6
        occ_blksize = min(nocc, max(1, int(np.floor(mem_avail * 0.6 / mem_occblk))))
        mem_auxblk = (occ_blksize * nvir + nao_pair) * dsize / 1e6
        aux_blksize = min(naux, max(1, int(np.floor(mem_avail * 0.3 / mem_auxblk))))
        log.debug('occ blksize for outcore ao2mo: %d/%d', occ_blksize, nocc)
        log.debug('aux blksize for outcore ao2mo: %d/%d', aux_blksize, naux)
        buf = np.empty(naux * occ_blksize * nvir, dtype=dtype)
        buf2 = np.empty(aux_blksize * occ_blksize * nvir, dtype=dtype)

        for i0, i1 in lib.prange(0, nocc, occ_blksize):
            nocci = i1 - i0
            ijslice = (i0, i1, nocc, nmo)
            p1 = 0
            OvL = np.ndarray((nocci, nvir, naux), dtype=dtype, buffer=buf)
            for Lpq in with_df.loop(blksize=aux_blksize):
                p0, p1 = p1, p1 + Lpq.shape[0]
                out = _ao2mo.nr_e2(Lpq, mo, ijslice, aosym='s2', out=buf2)
                OvL[:, :, p0:p1] = out.reshape(-1, nocci, nvir).transpose(1, 2, 0)
                Lpq = out = None
            ovL[i0:i1] = (
                OvL  # this avoids slow operations like ovL[i0:i1,:,p0:p1] = ...
            )
            OvL = None
        buf = buf2 = None

    return ovL


def _init_mp_df_eris_direct(
    with_df, occ_coeff, vir_coeff, max_memory, h5obj=None, log=None
):
    from pyscf import gto
    from pyscf.df.incore import fill_2c2e
    from pyscf.ao2mo.outcore import balance_partition
    from pyscf.ao2mo import _ao2mo

    if log is None:
        log = logger.new_logger(with_df)

    mol = with_df.mol
    auxmol = with_df.auxmol
    nbas = mol.nbas
    nao, nocc = occ_coeff.shape
    nvir = vir_coeff.shape[1]
    nmo = nocc + nvir
    nao_pair = nao * (nao + 1) // 2
    naoaux = auxmol.nao_nr()

    dtype = occ_coeff.dtype
    assert dtype == np.float64
    dsize = 8

    mo = np.asarray(np.hstack((occ_coeff, vir_coeff)), order='F')
    ijslice = (0, nocc, nocc, nmo)

    tspans = np.zeros((5, 2))
    tnames = ['j2c', 'j3c', 'xform', 'save', 'fit']
    tick = (logger.process_clock(), logger.perf_counter())
    # precompute for fitting
    j2c = fill_2c2e(mol, auxmol)
    try:
        m2c = scipy.linalg.cholesky(j2c, lower=True)
        tag = 'cd'
    except scipy.linalg.LinAlgError:
        e, u = np.linalg.eigh(j2c)
        cond = abs(e).max() / abs(e).min()
        keep = abs(e) > THRESH_LINDEP
        log.debug('cond(j2c) = %g', cond)
        log.debug('keep %d/%d cderi vectors', np.count_nonzero(keep), keep.size)
        e = e[keep]
        u = u[:, keep]
        m2c = lib.dot(u * e**-0.5, u.T.conj())
        tag = 'eig'
    j2c = None
    naux = m2c.shape[1]
    tock = (logger.process_clock(), logger.perf_counter())
    tspans[0] += np.asarray(tock) - np.asarray(tick)

    mem_avail = max_memory - lib.current_memory()[0]

    incore = h5obj is None
    if incore:
        ovL = np.empty((nocc, nvir, naoaux), dtype=dtype)
        mem_avail -= ovL.size * dsize / 1e6
    else:
        ovL_shape = (nocc, nvir, naux)
        ovL = h5obj.create_dataset(
            'ovL', ovL_shape, dtype=dtype, chunks=(1, *ovL_shape[1:])
        )
        h5tmp = lib.H5TmpFile()
        Lov0_shape = (naoaux, nocc, nvir)
        Lov0 = h5tmp.create_dataset(
            'Lov0', Lov0_shape, dtype=dtype, chunks=(1, *Lov0_shape[1:])
        )

    # buffer
    mem_blk = nao_pair * 2 * dsize / 1e6
    aux_blksize = max(1, min(naoaux, int(np.floor(mem_avail * 0.7 / mem_blk))))
    auxshl_range = balance_partition(auxmol.ao_loc, aux_blksize)
    auxlen = max([x[2] for x in auxshl_range])
    log.info(
        'mem_avail = %.2f  mem_blk = %.2f  auxlen = %d', mem_avail, mem_blk, auxlen
    )
    buf0 = np.empty(auxlen * nao_pair, dtype=dtype)
    buf0T = np.empty(auxlen * nao_pair, dtype=dtype)

    # precompute for j3c
    comp = 1
    aosym = 's2ij'
    int3c = gto.moleintor.ascint3(mol._add_suffix('int3c2e'))
    atm_f, bas_f, env_f = gto.mole.conc_env(
        mol._atm, mol._bas, mol._env, auxmol._atm, auxmol._bas, auxmol._env
    )
    ao_loc_f = gto.moleintor.make_loc(bas_f, int3c)
    cintopt = gto.moleintor.make_cintopt(atm_f, bas_f, env_f, int3c)

    def calc_j3c_ao(kshl0, kshl1):
        shls_slice = (0, nbas, 0, nbas, nbas + kshl0, nbas + kshl1)
        pqL = gto.moleintor.getints3c(
            int3c,
            atm_f,
            bas_f,
            env_f,
            shls_slice,
            comp,
            aosym,
            ao_loc_f,
            cintopt,
            out=buf0,
        )
        Lpq = lib.transpose(pqL, out=buf0T)
        pqL = None
        return Lpq

    # transform
    k1 = 0
    for auxshl_rg in auxshl_range:
        kshl0, kshl1, dk = auxshl_rg
        k0, k1 = k1, k1 + dk
        log.debug(
            'kshl = [%d:%d/%d]  [%d:%d/%d]', kshl0, kshl1, auxmol.nbas, k0, k1, naoaux
        )
        tick = (logger.process_clock(), logger.perf_counter())
        lpq = calc_j3c_ao(kshl0, kshl1)
        tock = (logger.process_clock(), logger.perf_counter())
        tspans[1] += np.asarray(tock) - np.asarray(tick)
        lov = _ao2mo.nr_e2(lpq, mo, ijslice, aosym='s2', out=buf0)
        tick = (logger.process_clock(), logger.perf_counter())
        tspans[2] += np.asarray(tick) - np.asarray(tock)
        if incore:
            ovl = lib.transpose(lov, out=buf0T).reshape(nocc, nvir, dk)
            ovL[:, :, k0:k1] = ovl
            ovl = None
        else:
            Lov0[k0:k1] = lov.reshape(dk, nocc, nvir)
        lpq = lov = None
        tock = (logger.process_clock(), logger.perf_counter())
        tspans[3] += np.asarray(tock) - np.asarray(tick)
    buf0 = buf0T = None

    tick = (logger.process_clock(), logger.perf_counter())
    # fit
    if tag == 'cd':
        drv = getattr(libmp, 'trisolve_parallel_grp', None)
    if incore:
        if tag == 'cd':
            if drv is None:
                ovL = scipy.linalg.solve_triangular(
                    m2c, ovL.T, lower=True, overwrite_b=True, check_finite=False
                ).T
            else:
                grpfac = 10
                drv(
                    m2c.ctypes.data_as(ctypes.c_void_p),
                    ovL.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(naux),
                    ctypes.c_int(nocc * nvir),
                    ctypes.c_int(grpfac),
                )
        else:
            nvxao = nvir * naoaux
            nvx = nvir * naux
            mem_blk = nvx * dsize / 1e6
            occ_blksize = max(1, min(nocc, int(np.floor(mem_avail * 0.5 / mem_blk))))
            buf = np.empty(occ_blksize * nvx, dtype=dtype)
            ovL = ovL.reshape(-1)
            for i0, i1 in lib.prange(0, nocc, occ_blksize):
                nocci = i1 - i0
                out = np.ndarray((nocci * nvir, naux), dtype=dtype, buffer=buf)
                lib.dot(
                    ovL[i0 * nvxao:i1 * nvxao].reshape(nocci * nvir, naoaux),
                    m2c,
                    c=out,
                )
                ovL[i0 * nvx:i1 * nvx] = out.reshape(-1)
            ovL = ovL[:nocc * nvx].reshape(nocc, nvir, naux)
            buf = None
    else:
        nvxao = nvir * naoaux
        nvx = nvir * naux
        mem_blk = nvxao * dsize / 1e6
        occ_blksize = max(1, min(nocc, int(np.floor(mem_avail * 0.4 / mem_blk))))
        for i0, i1 in lib.prange(0, nocc, occ_blksize):
            nocci = i1 - i0
            ivL = np.asarray(Lov0[:, i0:i1].transpose(1, 2, 0), order='C')
            if tag == 'cd':
                if drv is None:
                    ivL = scipy.linalg.solve_triangular(
                        m2c, ivL.T, lower=True, overwrite_b=True, check_finite=False
                    ).T
                else:
                    grpfac = 10
                    drv(
                        m2c.ctypes.data_as(ctypes.c_void_p),
                        ivL.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(naux),
                        ctypes.c_int(nocci * nvir),
                        ctypes.c_int(grpfac),
                    )
            else:
                ivL = lib.dot(ivL.reshape(nocci * nvir, naoaux), m2c).reshape(
                    nocci, nvir, naux
                )
            ovL[i0:i1] = ivL
        del h5tmp['Lov0']
        h5tmp.close()
        Lov0 = None
    tock = (logger.process_clock(), logger.perf_counter())
    tspans[4] += np.asarray(tock) - np.asarray(tick)

    for tspan, tname in zip(tspans, tnames):
        log.debug(
            'ao2mo CPU time for %-10s  %9.2f sec  wall time %9.2f sec', tname, *tspan
        )
    log.info('')

    return ovL
