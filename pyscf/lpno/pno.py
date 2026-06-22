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

from pyscf.lpno import tools

from pyscf import lib
einsum = lib.einsum


def get_pno1(vij, tij, ev, thresh=None, pct_occ=None, thresh_ene=0.9, ieqj=False, return_occ=False):
    tij_tld = 2*tij - tij.T
    dij = (lib.dot(tij_tld.T.conj(), tij) + lib.dot(tij_tld, tij.T.conj()))
    if ieqj:
        dij *= 0.5
    n, v = np.linalg.eigh(dij)
    ntot = n.size

    n = abs(n)
    order = np.argsort(n)[::-1]
    n = n[order]
    v = v[:, order]
    if thresh is not None:
        nkeep = np.count_nonzero(n > thresh)
    elif pct_occ is not None:
        pct_occ_sum = np.cumsum(n/np.sum(n))
        nkeep = np.count_nonzero(pct_occ_sum < pct_occ)
    else:
        raise ValueError

    Epsv_d = 2 * einsum('ab,ab->', vij, tij).real

    # Increase nkeep and refine Epno by energy criterion
    if thresh_ene is not None and nkeep < ntot:

        vij_pno = reduce(lib.dot, (v.T.conj(), vij, v))
        tij_pno = reduce(lib.dot, (v.T, tij, v.conj()))
        Epno_d = 2 * einsum('ab,ab->', vij_pno[:nkeep, :nkeep], tij_pno[:nkeep, :nkeep]).real

        if Epno_d/Epsv_d < thresh_ene:
            nkeep0 = nkeep
            for nkeep in range(nkeep0+1, ntot):
                dEpno_d = np.dot(vij_pno[nkeep-1, :nkeep], tij_pno[nkeep-1, :nkeep])
                dEpno_d += np.dot(vij_pno[:nkeep, nkeep-1], tij_pno[:nkeep, nkeep-1])
                dEpno_d -= vij_pno[nkeep-1, nkeep-1] * tij_pno[nkeep-1, nkeep-1]
                dEpno_d *= 2

                Epno_d += dEpno_d.real
                if abs(Epno_d/Epsv_d) >= thresh_ene:
                    break

    vact = v[:, :nkeep]
    eact, vact = tools.pseudocano(ev, vact)

    if return_occ:
        return eact, vact, n[:nkeep].copy(), n.copy()
    return eact, vact
