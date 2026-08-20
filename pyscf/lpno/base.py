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

"""Shared base class for projection-based local MP2 methods (molecular)."""

import numpy as np
from functools import reduce

from pyscf.mp.mp2 import MP2
from pyscf import df
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__

from pyscf.lpno import tools, ao2mo

einsum = lib.einsum


def kernel(mp, pair_domain):
    log = logger.new_logger(mp)

    cput = (logger.process_clock(), logger.perf_counter())
    converged = False
    t2 = mp.init_guess(pair_domain)
    e_corr_last = 0
    e_corr = mp.energy_corr(pair_domain, t2)
    log.info('Init E_corr= %.10g', e_corr)
    cput = log.timer_debug1('Init guess', *cput)
    for cycle in range(1, mp.max_cycle + 1):
        t2new = mp.update_amp(t2, pair_domain)
        if mp.iterative_damping < 1.0:
            t2new_data = mp.iterative_damping * t2new.data + (1 - mp.iterative_damping) * t2.data
            t2new.data = t2new_data
        t2, t2new = t2new, None
        e_corr_last, e_corr = e_corr, mp.energy_corr(pair_domain, t2)
        delta_e = e_corr - e_corr_last
        log.info('cycle= %d  E_corr= %.15g  delta_E= %.9g',
                 cycle, e_corr, delta_e)
        cput = log.timer_debug1('%s cycle' % (mp.__class__.__name__), *cput)
        if abs(delta_e) < mp.conv_tol:
            converged = True
            break

    return e_corr, t2, converged


class BASE(MP2):
    '''Base class for projection-based local MP2 methods.

    Args:
        lo_coeff : localized occupied MO coefficients
        thresh : overall accuracy control
        frozen : int or list of frozen core orbitals
    '''
    max_cycle = getattr(__config__, 'lpno_max_cycle', 50)
    conv_tol = getattr(__config__, 'lpno_conv_tol', 1e-8)
    # iterative_damping: fraction of new amplitudes mixed per iteration.
    # Default 1.0 means no damping (pure Jacobi update).  Only active when
    # DIIS is not used; reduce below 1.0 to stabilise slow-converging cases.
    iterative_damping = getattr(__config__, 'lpno_iterative_damping', 1.0)
    conv_tol_normt = getattr(__config__, 'lpno_conv_tol_normt', 1e-4)

    incore_complete = getattr(__config__, 'lpno_incore_complete', False)

    OSV = None
    PairDomain = None

    def __init__(self, mf, lo_coeff, frozen=None):
        MP2.__init__(self, mf, frozen=frozen)

        log = logger.new_logger(self)

        self.lo_coeff = lo_coeff
        if getattr(self._scf, 'with_df', None) is not None:
            self.with_df = self._scf.with_df
            log.info('Using with_df from SCF object')
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
            log.info('Using auxbasis: %s', self.with_df.auxbasis)

        self.mo_energy = mf.mo_energy

        self.e_pair = None
        self.e_pair_ss = None
        self.e_pair_os = None
        self._e_pair = None
        self._e_pair_ss = None
        self._e_pair_os = None

        self._pair_mask = None
        self.converged = False
        self.t2 = None

        self.thresh_distpair = 1e-6
        self.rmin_distpair = 4  # Bohr
        self.thresh_foo = 0
        self._s1e = None
        self._foo = None
        self.force_outcore = False
        self.chkfile = None

        self._keys.update(['lo_coeff', 'thresh_osv', 'with_df', 'mo_energy',
                           'max_cycle', 'conv_tol', 'converged', '_s1e', '_foo',
                           'thresh_distpair', 'rmin_distpair',
                           'OSV', 'PairDomain', 'thresh_foo',
                           'e_pair', 'e_pair_ss', 'e_pair_os',
                           '_e_pair', '_e_pair_ss', '_e_pair_os',
                           '_pair_mask'])

    @property
    def pair_mask(self):
        return self._pair_mask['Strong pair']

    @property
    def _e_corr(self):
        e = {k: v.sum() for k, v in self._e_pair.items()}
        e['Total'] = sum([v for k, v in e.items() if k != 'Total'])
        return e

    @property
    def _e_corr_ss(self):
        e = {k: v.sum() for k, v in self._e_pair_ss.items()}
        e['Total'] = sum([v for k, v in e.items() if k != 'Total'])
        return e

    @property
    def _e_corr_os(self):
        e = {k: v.sum() for k, v in self._e_pair_os.items()}
        e['Total'] = sum([v for k, v in e.items() if k != 'Total'])
        return e

    @property
    def s1e(self):
        if self._s1e is None:
            self._s1e = self._scf.get_ovlp()
        return self._s1e

    @property
    def foo(self):
        if self._foo is None:
            occ_coeff = self.split_mo_coeff()[1]
            occ_energy = self.split_mo_energy()[1]
            u = reduce(lib.dot, (self.lo_coeff.T.conj(), self.s1e, occ_coeff))
            self._foo = lib.dot(u * occ_energy, u.T.conj())
        return self._foo

    def dump_flags(self, verbose=None):
        MP2.dump_flags(self, verbose)
        log = logger.new_logger(self, verbose)
        log.info('OSV = %s', self.OSV)
        log.info('PairDomain = %s', self.PairDomain)
        log.info('chkfile = %s', self.chkfile)
        log.info('max_cycle = %d', self.max_cycle)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('thresh_distpair = %g', self.thresh_distpair)
        log.info('rmin_distpair = %g', self.rmin_distpair)
        log.info('thresh_foo = %g', self.thresh_foo)

    def kernel(self, eris=None):

        log = logger.new_logger(self)
        cput0 = (logger.process_clock(), logger.perf_counter())

        self.dump_flags()

        if self.chkfile:
            lib.chkfile.dump(self.chkfile, 'lo_coeff', self.lo_coeff)

        # FIXME: be consistent with MP2
        self.e_hf = self._scf.e_tot

        timer = {}
        pair_domain = self.make_pair_domain(eris=eris, timer=timer)

        cput1 = (logger.process_clock(), logger.perf_counter())

        self.e_corr, self.t2, self.converged = kernel(self, pair_domain)

        cput2 = log.timer('Residual eqn', *cput1)
        timer['Residual eqn'] = np.asarray(cput2) - np.asarray(cput1)
        timer['Total'] = sum([v for k, v in timer.items()])

        self.e_corr_ss = getattr(self.e_corr, 'e_corr_ss', 0)
        self.e_corr_os = getattr(self.e_corr, 'e_corr_os', 0)
        self.e_corr = float(self.e_corr)

        log.timer(self.__class__.__name__, *cput0)

        self._timer = timer
        self._finalize(timer)

    def _finalize(self, timer):
        self._energy_finalize()
        self._timer_summary(timer)
        self._ecomp_summary(self._e_corr)

    def _energy_finalize(self):
        MP2._finalize(self)

    def _timer_summary(self, timer, verbose=None):
        log = logger.new_logger(self, verbose)

        sep = '-' * 62
        log.info('')
        log.info(sep)
        log.info('CPU and wall time (sec) breakdown')
        log.info(sep)
        ks = _get_sorted_keys(timer)
        for k in ks:
            tcpu, twall = timer[k]
            rcpu, rwall = timer[k] / timer['Total'] * 100
            log.info('%-15s  %9.2f (%6.2f%%)  %9.2f (%6.2f%%)', k, tcpu, rcpu, twall, rwall)
        log.info(sep)
        log.info('')

    def _ecomp_summary(self, ecomp, verbose=None):
        log = logger.new_logger(self, verbose)

        sep = '-' * 44
        log.info(sep)
        log.info('Correlation energy breakdown')
        log.info(sep)
        ks = _get_sorted_keys(ecomp)
        for k in ks:
            e = ecomp[k]
            r = e / ecomp['Total'] * 100
            log.info('%-15s  % 14.9f (%6.2f%%)', k, e, r)
        log.info(sep)
        log.info('')

    def _update_pair_mask(self, k, v):
        if self._pair_mask is None:
            self._pair_mask = {}
        self._pair_mask[k] = tools._to_full(v, self.nocc)

    def _update_pair_energy(self, k, v, mask=None):
        if mask is not None:
            if self.e_pair is None:
                self.e_pair = np.zeros(self.nocc * (self.nocc + 1) // 2, dtype=np.float64)
                self.e_pair_ss = np.zeros_like(self.e_pair)
                self.e_pair_os = np.zeros_like(self.e_pair)
            mask = tools._to_tril(mask, self.nocc)
            self.e_pair[mask] = v[mask]
            if getattr(v, 'e_corr_ss', None) is not None:
                self.e_pair_ss[mask] = v.e_corr_ss[mask]
                self.e_pair_os[mask] = v.e_corr_os[mask]

        if self._e_pair is None:
            self._e_pair = {}
            self._e_pair_ss = {}
            self._e_pair_os = {}
        self._e_pair_ss[k] = getattr(v, 'e_corr_ss', 0)
        self._e_pair_os[k] = getattr(v, 'e_corr_os', 0)
        self._e_pair[k] = np.asarray(v)

    def energy_corr(self, pair_domain, t2):
        self._energy_corr_pair(pair_domain, t2)

        e_corr = lib.tag_array(self.e_pair.sum(),
                               e_corr_ss=self.e_pair_ss.sum(),
                               e_corr_os=self.e_pair_os.sum())

        return e_corr

    def _energy_corr_pair(self, pair_domain, t2, with_ex=True):
        nocc = self.nocc
        e_corr_ss = np.zeros((nocc, nocc))
        e_corr_os = np.zeros((nocc, nocc))
        for ipair, (i, j) in enumerate(pair_domain.loop_pair()):
            fac = 1 if i == j else 2
            ed = einsum('ab,ab->', t2[i, j], pair_domain.K[i, j]) * fac
            if with_ex:
                ex = einsum('ab,ab->', t2[i, j], pair_domain.J[i, j]) * fac
            else:
                ex = 0
            e_corr_ss[i, j] = ed - ex
            e_corr_os[i, j] = ed
        e_corr_ss = lib.pack_tril(e_corr_ss)
        e_corr_os = lib.pack_tril(e_corr_os)
        e_corr = lib.tag_array(e_corr_ss + e_corr_os, e_corr_ss=e_corr_ss, e_corr_os=e_corr_os)

        self._update_pair_energy('Strong pair', e_corr, pair_domain.pair_mask)

        if self.chkfile:
            lib.chkfile.dump(self.chkfile, 'pair/Estrong', e_corr)

        return e_corr

    def _energy_corr_pair_sc_mp2(self, pair_domain):
        t2 = self.init_guess(pair_domain)
        return self._energy_corr_pair(pair_domain, t2)

    def init_guess(self, pair_domain):
        pass

    def update_amp(self, t2, pair_domain):
        pass

    def get_normt(self, t2, t2new):
        return np.linalg.norm(t2.data - t2new.data)

    def get_split_mask(self):
        maskact = self.get_frozen_mask()
        maskocc = self.mo_occ > 1e-10
        return (
            ~maskact & maskocc,
            maskact & maskocc,
            maskact & ~maskocc,
            ~maskact & ~maskocc
        )

    def split_mo_coeff(self):
        return [self.mo_coeff[:, m] for m in self.get_split_mask()]

    def split_mo_energy(self):
        return [self.mo_energy[m] for m in self.get_split_mask()]

    def check_sanity(self):
        log = logger.new_logger(self)
        occ_coeff = self.split_mo_coeff()[1]
        if not _is_unitary_rotation(occ_coeff, self.lo_coeff, self.s1e):
            log.error('localized orbitals are not unitary rotations of active occupied orbitals.')
            raise RuntimeError

    def make_pair_domain(self, eris=None, timer=None):
        pass

    def screen_dist_pair(self, osv, pair_mask=None):
        '''
            The following attributes are set by this function:
                - _e_pair/_ss/_os['Dist pair']
                - _pair_mask['Dist pair']
        '''
        log = logger.new_logger(self)
        if self.mol.__dict__.get('pbc_intor', None):
            log.warn('skip pair screening for pbc calculations')
            return np.ones((self.nocc,) * 2, dtype=bool), 0.

        mol = self.mol
        lo_coeff = self.lo_coeff
        vir_coeff = self.split_mo_coeff()[2]
        moe_lo = np.diag(self.foo)

        epair, Rpair = pair_energy_dipole(mol, lo_coeff, moe_lo, vir_coeff, osv)
        if pair_mask is None:
            pair_mask = Rpair <= self.rmin_distpair
            pair_mask[abs(epair) > self.thresh_distpair] = True
        epair[pair_mask] = 0
        epair = lib.tag_array(epair, e_corr_ss=epair * 0.5, e_corr_os=epair * 0.5)

        nnear = np.count_nonzero(pair_mask)
        ndist = np.count_nonzero(~pair_mask)
        log.debug('Dist-pair prescreen: %d Near | %d Dist | %d Total pairs',
                  nnear, ndist, nnear + ndist)

        if self.chkfile:
            lib.chkfile.dump(self.chkfile, 'pair/mask_dist', pair_mask)
            lib.chkfile.dump(self.chkfile, 'pair/Edist', epair)
            lib.chkfile.dump(self.chkfile, 'pair/R', Rpair)

        pair_mask = lib.unpack_tril(pair_mask)

        return pair_mask, epair

    ao2mo = ao2mo.get_eris


def pair_energy_dipole(mol, lo_coeff, moe_lo, vir_coeff, osv):
    nocc = lo_coeff.shape[1]
    dipao = mol.intor_symmetric('int1e_r', comp=3)
    lo_r = einsum('xpq,pi,qi->ix', dipao, lo_coeff.conj(), lo_coeff)
    Rvecpair = lo_r[:, None, :] - lo_r
    Rpair = np.linalg.norm(Rvecpair, axis=-1)
    diplv = einsum('xpq,pi,qa->ixa', dipao, lo_coeff.conj(), vir_coeff)
    diplosv = [einsum('xa,aA->xA', diplv[i], osv.u[i]) for i in range(nocc)]
    epair = np.zeros((nocc, nocc))
    for i in range(nocc):
        for j in range(i):
            Rbar = Rvecpair[i, j] / Rpair[i, j]
            vab = einsum('xA,xB->AB', diplosv[i], diplosv[j])
            vab -= einsum('A,B->AB', np.dot(Rbar, diplosv[i]), np.dot(Rbar, diplosv[j])) * 3
            tab = vab.conj() / (moe_lo[i] + moe_lo[j] - (osv.e[i][:, None] + osv.e[j]))
            eij = einsum('ab,ab->', tab, vab) / Rpair[i, j]**6 * 4
            epair[i, j] = epair[j, i] = eij
    epair += epair.T    # off diag is multiplied by 2

    return lib.pack_tril(epair), lib.pack_tril(Rpair)


def _is_unitary_rotation(c1, c2, s, tol=1e-6):
    u = reduce(lib.dot, (c1.T.conj(), s, c2))
    err1 = abs(lib.dot(u.T.conj(), u) - np.eye(u.shape[1])).max()
    err2 = abs(lib.dot(u, u.T.conj()) - np.eye(u.shape[0])).max()
    return err1 < tol and err2 < tol


def _get_sorted_keys(d, exclude='Total'):
    ks = []
    vs = []
    for k, v in d.items():
        if exclude:
            if k == exclude:
                continue
        ks.append(k)
        if isinstance(v, (list, np.ndarray)):
            vs.append(abs(v[0]))
        else:
            vs.append(abs(v))
    ks = np.asarray(ks)[np.argsort(vs)[::-1]].tolist()
    if exclude:
        ks += [exclude]
    return ks
