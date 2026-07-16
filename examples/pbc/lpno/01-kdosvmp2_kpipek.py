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

'''kdOSVMP2 with KPM localization and frozen core.

NOTE: this example needs periodic Pipek-Mezey (pyscf.pbc.lo.kpipek), a
localization code developed by Hong-Zhou Ye and Gengzhi Yang.
For a stock-pyscf example, see 00-kpnomp2.py.
'''

import argparse
import numpy as np
from pyscf.pbc import gto, scf, mp as pbcmp

from pyscf.pbc.lpno import kdOSVMP2
from pyscf.pbc.lpno.kpts2supcell import k2s_scf, k2s_aoint
from pyscf.lpno.tools import guess_frozen, sort_orb_by_cell


def _require_kpipek():
    try:
        from pyscf.pbc.lo import kpipek
    except ModuleNotFoundError as err:
        raise RuntimeError('KPM example requires pyscf.pbc.lo.kpipek') from err
    return kpipek


def make_cell(system):
    '''Build a PySCF Cell for the given system name.'''
    if system == 'lih':
        cell = gto.Cell(
            atom='Li 0 0 0; H 2.042 2.042 2.042',
            a='''0     2.042 2.042
                 2.042 0     2.042
                 2.042 2.042 0''',
            basis='pobtzvp', verbose=4,
            precision=1e-14, max_memory=320000, exp_to_discard=0.1,
        )
    elif system == 'mgo':
        cell = gto.Cell(
            atom='Mg 0 0 0; O 2.1695 2.1695 2.1695',
            a='''0      2.1695 2.1695
                 2.1695 0      2.1695
                 2.1695 2.1695 0''',
            basis='pobtzvp', verbose=4,
            precision=1e-14, max_memory=320000, exp_to_discard=0.1,
        )
    elif system == 'water':
        cell = gto.Cell(
            atom='''O  0.000  0.000  0.117
                    H  0.000  0.757 -0.469
                    H  0.000 -0.757 -0.469''',
            a=np.eye(3) * 10.0,
            basis='def2-svp', verbose=4,
        )
    else:
        raise ValueError(f'Unknown system: {system}')

    cell.build()
    return cell


def run_kdosvmp2(cell, kmesh, frozen_per_kpt=None, thresh_osv=1e-5):
    '''Run kdOSVMP2 with KPM localization and return ``(mmp, kmp)``.'''
    kpipek = _require_kpipek()
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

    mf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='weigend')
    mf.kernel()

    if frozen_per_kpt is None:
        frozen_per_kpt = guess_frozen(cell)
    nocc = cell.nelectron // 2
    nVband = nocc - frozen_per_kpt
    kfrozen = frozen_per_kpt * nkpts

    print(f'\nFrozen core: {frozen_per_kpt} per k-point, {kfrozen} total (supercell)')
    print(f'Active valence: {nVband} per k-point\n')

    mo_val = np.asarray([mf.mo_coeff[k][:, frozen_per_kpt:nocc]
                         for k in range(nkpts)])
    mlo = kpipek.KPM(cell, mo_val, kpts)
    mlo.verbose = 4
    mlo.kernel()
    while True:
        mo, stable = mlo.stability(return_status=True)
        if stable:
            break
        mlo.kernel(mo)

    orbloc = mlo.get_wannier_function()
    im_norm = np.linalg.norm(orbloc.imag)
    if im_norm > 1e-4:
        print(f'WARNING: Large imaginary norm in Wannier functions: {im_norm:.6e}')
    orbloc = orbloc.real

    supcell_mf = k2s_scf(mf, kmesh=kmesh)
    scell = supcell_mf.cell
    s1e = k2s_aoint(cell, kpts, mf.get_ovlp())
    orbloc_sorted = sort_orb_by_cell(scell, orbloc, nkpts, s=s1e)

    mmp = kdOSVMP2(mf, orbloc_sorted, kmesh=kmesh, frozen=kfrozen).set(verbose=4)
    mmp.thresh_osv = thresh_osv
    mmp.kernel()

    kmp = pbcmp.KMP2(mf, frozen=frozen_per_kpt)
    kmp.verbose = 5
    kmp.kernel()

    de = mmp.e_corr - kmp.e_corr
    pct = abs(de) / abs(kmp.e_corr) * 100
    print(f'\nKMP2 E_corr    = {kmp.e_corr:.10f}')
    print(f'kdOSVMP2 E_corr = {mmp.e_corr:.10f}')
    print(f'Error           = {de:+.10f} ({pct:.2f}%)')

    return mmp, kmp


def parse_args():
    parser = argparse.ArgumentParser(
        description='kdOSVMP2 with KPM localization and frozen core.')
    parser.add_argument('system', choices=['lih', 'mgo', 'water'],
                        help='System to run')
    parser.add_argument('-k', '--kmesh', default='2,2,2',
                        help='k-mesh as comma-separated ints (default: 2,2,2)')
    parser.add_argument('--thresh', type=float, default=1e-5,
                        help='thresh_osv (default: 1e-5)')
    return parser.parse_args()


def parse_kmesh(s):
    '''Parse kmesh from "2,2,2" or "222" format.'''
    if ',' in s:
        return [int(x) for x in s.split(',')]
    return [int(c) for c in s]


if __name__ == '__main__':
    args = parse_args()
    kmesh = parse_kmesh(args.kmesh)
    cell = make_cell(args.system)
    run_kdosvmp2(cell, kmesh, thresh_osv=args.thresh)
