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

"""Build periodic DF ERIs in the occupied-virtual block."""

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.pbc.lpno.kpts2supcell import K2SDF


def get_eris(mylpno, vir_coeff=None, ovL=None, ovL_to_save=None):
    """Build the periodic DF ERI helper used by LPnO methods (PBC only)."""
    log = logger.new_logger(mylpno)

    with_df = getattr(mylpno, 'with_df', None)
    assert with_df is not None
    assert hasattr(with_df, 'kpts'), 'get_eris in pyscf.pbc.lpno.ao2mo requires a PBC DF object'

    lo_coeff = mylpno.lo_coeff
    if vir_coeff is None:
        vir_coeff = mylpno.split_mo_coeff()[2]

    pbc_ao2mo_mode = getattr(mylpno, 'pbc_ao2mo_mode', 'incore')
    nlo_per_cell = getattr(mylpno, 'nlo_per_cell', None)
    kscf = getattr(mylpno, '_kscf', None)
    kmesh = getattr(mylpno, '_kmesh', None)
    log.debug('Building PBC DF from K2SDF (mode=%s)', pbc_ao2mo_mode)

    eris = _DFINCOREERIS_PBC(
        with_df,
        lo_coeff,
        vir_coeff,
        mylpno.max_memory,
        mode=pbc_ao2mo_mode,
        nlo_per_cell=nlo_per_cell,
        kscf=kscf,
        kmesh=kmesh,
        verbose=log.verbose,
        stdout=log.stdout,
    )
    eris.build()
    return eris


class _DFINCOREERIS_PBC(K2SDF):
    def __init__(
        self,
        with_df,
        occ_coeff,
        vir_coeff,
        max_memory,
        mode='incore',
        nlo_per_cell=None,
        kscf=None,
        kmesh=None,
        verbose=None,
        stdout=None,
    ):
        K2SDF.__init__(self, with_df, kmesh=kmesh)
        self.occ_coeff = occ_coeff
        self.vir_coeff = vir_coeff
        self.max_memory = max_memory
        self.verbose = verbose
        self.stdout = stdout
        self.ovLR = None
        self.ovLI = None
        self._mode = mode
        self._nlo_per_cell = nlo_per_cell
        self._kscf = kscf
        self._ovL_k_R = None
        self._ovL_k_I = None
        self._U_k = None
        self._phase_factor = None

    @property
    def nocc(self):
        return self.occ_coeff.shape[1]

    @property
    def nvir(self):
        return self.vir_coeff.shape[1]

    def build(self):
        """Build the periodic DF ERI tensors using the selected mode.

        Supported modes:
            ``'incore'`` (default): build full supercell ``ovLR``/``ovLI`` in
                memory, with an automatic fall-back to ``'outcore'`` when
                memory is insufficient.
            ``'outcore'``: build supercell ``ovLR``/``ovLI`` on disk via an
                HDF5 scratch file.
        """
        log = logger.new_logger(self)
        cput1 = (logger.process_clock(), logger.perf_counter())
        if self._mode == 'outcore':
            if self._nlo_per_cell is not None and self._kscf is not None:
                return self._build_outcore(log, cput1)
        elif self._mode == 'incore':
            if self._nlo_per_cell is not None and self._kscf is not None:
                mem_df = 2 * self.nocc * self.nvir * self.Naux_ibz * 8 / 1024**2
                mem_avail = self.max_memory - lib.current_memory()[0]
                if mem_df > mem_avail:
                    log.info(
                        'Insufficient memory for incore (%.0f MB > %.0f MB avail), '
                        'switching to outcore',
                        mem_df,
                        mem_avail,
                    )
                    return self._build_outcore(log, cput1)
                return self._build_incore(log, cput1)
        else:
            raise ValueError(
                f"unknown pbc_ao2mo_mode {self._mode!r}; choose 'incore' or 'outcore'"
            )

    def get_occ_blk(self, i0, i1):
        if (
            getattr(self, '_refcell_ovLR', None) is not None
            and i1 <= self._nlo_per_cell
        ):
            return self._refcell_ovLR[i0:i1], self._refcell_ovLI[i0:i1]
        if self.ovLR is not None:
            return np.asarray(self.ovLR[i0:i1], order='C'), np.asarray(
                self.ovLI[i0:i1], order='C'
            )
        return self._get_occ_blk_slow(i0, i1)

    def close(self):
        """Release outcore resources."""
        self.ovLR = None
        self.ovLI = None
        if getattr(self, '_refcell_ovLR', None) is not None:
            self._refcell_ovLR = None
            self._refcell_ovLI = None
        if getattr(self, '_h5file', None) is not None:
            self._h5file.close()
            self._h5file = None

    def _get_occ_blk_slow(self, i0, i1):
        npc = self._nlo_per_cell
        nkpts = len(self.kpts)
        nvir = self.nvir
        Naux = self.Naux_ibz
        nqpts = len(self._qi_ranges)
        blkR = np.zeros((i1 - i0, nvir, Naux))
        blkI = np.zeros((i1 - i0, nvir, Naux))
        for idx, j in enumerate(range(i0, i1)):
            j_ref = j % npc
            R_idx = j // npc
            for qi in range(nqpts):
                b0, nauxq = self._qi_ranges[qi]
                for ki in range(nkpts):
                    kj = self._kj_for_ki_qi[ki][qi]
                    pf = self._phase_factor[R_idx, ki]  # complex scalar
                    ovkR = self._ovL_k_R[ki][
                        j_ref, :, b0 : b0 + nauxq
                    ]  # (nvir_k, nauxq)
                    ovkI = self._ovL_k_I[ki][j_ref, :, b0 : b0 + nauxq]
                    U_kj = self._U_k[kj]  # (nvir_k, nvir), complex
                    UR = U_kj.real.T
                    UI = U_kj.imag.T
                    tmpR = lib.dot(UR, ovkR) - lib.dot(UI, ovkI)
                    tmpI = lib.dot(UI, ovkR) + lib.dot(UR, ovkI)
                    blkR[idx, :, b0 : b0 + nauxq] += pf.real * tmpR - pf.imag * tmpI
                    blkI[idx, :, b0 : b0 + nauxq] += pf.real * tmpI + pf.imag * tmpR
        return blkR, blkI

    def _project_blk(self, j, proj):
        """Return ``proj.T`` contracted with the occupied block ``j``."""
        if self._ovL_k_R is None:
            ivLR, ivLI = self.get_occ_blk(j, j + 1)
            return lib.dot(proj.T.conj(), ivLR[0]), lib.dot(proj.T.conj(), ivLI[0])

        npc = self._nlo_per_cell
        nkpts = len(self.kpts)
        nproj = proj.shape[1]
        j_ref = j % npc
        R_idx = j // npc

        pk_R = [lib.dot(proj.conj().T, self._U_k[kj].real.T) for kj in range(nkpts)]
        pk_I = [lib.dot(proj.conj().T, self._U_k[kj].imag.T) for kj in range(nkpts)]

        return self._contract_proj_blk(j_ref, R_idx, pk_R, pk_I, nproj)

    def get_projected_blk(self, j, proj):
        """Public wrapper for projected occupied-virtual DF blocks."""
        return self._project_blk(j, proj)

    def get_projected_blk_batch(self, j, projectors):
        """Batch projected occupied-virtual DF blocks for one occupied index."""
        if not projectors:
            return []

        nproj_list = [proj.shape[1] for proj in projectors]
        total_nproj = sum(nproj_list)
        if total_nproj == 0:
            return [(np.zeros((0, self.Naux_ibz)), np.zeros((0, self.Naux_ibz)))
                    for _ in projectors]

        stacked_proj = np.hstack(projectors)
        if self._ovL_k_R is None:
            ivLR, ivLI = self.get_occ_blk(j, j + 1)
            resultR = lib.dot(stacked_proj.T.conj(), ivLR[0])
            resultI = lib.dot(stacked_proj.T.conj(), ivLI[0])
        else:
            npc = self._nlo_per_cell
            nkpts = len(self.kpts)
            j_ref = j % npc
            R_idx = j // npc
            pk_R = [
                lib.dot(stacked_proj.conj().T, self._U_k[kj].real.T)
                for kj in range(nkpts)
            ]
            pk_I = [
                lib.dot(stacked_proj.conj().T, self._U_k[kj].imag.T)
                for kj in range(nkpts)
            ]
            resultR, resultI = self._contract_proj_blk(
                j_ref, R_idx, pk_R, pk_I, total_nproj)

        results = []
        offset = 0
        for nproj in nproj_list:
            results.append(
                (resultR[offset:offset + nproj],
                 resultI[offset:offset + nproj]))
            offset += nproj
        return results

    def _contract_proj_blk(self, j_ref, R_idx, pk_R, pk_I, nproj):
        """Shared contraction kernel for projected DF blocks."""
        nkpts = len(self.kpts)
        Naux = self.Naux_ibz
        nqpts = len(self._qi_ranges)
        resultR = np.zeros((nproj, Naux))
        resultI = np.zeros((nproj, Naux))

        for qi in range(nqpts):
            b0, nauxq = self._qi_ranges[qi]
            kj_list = [self._kj_for_ki_qi[ki][qi] for ki in range(nkpts)]

            ovk_sR = np.ascontiguousarray(self._ovL_k_R[:, j_ref, :, b0 : b0 + nauxq])
            ovk_sI = np.ascontiguousarray(self._ovL_k_I[:, j_ref, :, b0 : b0 + nauxq])

            pk_sR = np.stack([pk_R[kj_list[ki]] for ki in range(nkpts)])
            pk_sI = np.stack([pk_I[kj_list[ki]] for ki in range(nkpts)])

            bR = np.matmul(pk_sR, ovk_sR) - np.matmul(pk_sI, ovk_sI)
            bI = np.matmul(pk_sI, ovk_sR) + np.matmul(pk_sR, ovk_sI)

            pf = self._phase_factor[R_idx, :]
            resultR[:, b0 : b0 + nauxq] += np.einsum(
                'k,kpa->pa', pf.real, bR
            ) - np.einsum('k,kpa->pa', pf.imag, bI)
            resultI[:, b0 : b0 + nauxq] += np.einsum(
                'k,kpa->pa', pf.real, bI
            ) + np.einsum('k,kpa->pa', pf.imag, bR)

        return resultR, resultI

    def _project_osv_blk_batch(self, occ_idx, components_list):
        """Batch ``get_projected_blk_from_osv`` for a shared occupied index."""
        npc = self._nlo_per_cell
        nkpts = len(self.kpts)
        j_ref = occ_idx % npc
        R_idx = occ_idx // npc
        nvir_k = self._osv_cache_R[0][0].shape[1]

        nproj_list = [comp[0][1].shape[1] for comp in components_list]
        total_nproj = sum(nproj_list)

        pk_R = [None] * nkpts
        pk_I = [None] * nkpts
        for kj in range(nkpts):
            pkR = np.empty((total_nproj, nvir_k))
            pkI = np.empty((total_nproj, nvir_k))
            offset = 0
            for components in components_list:
                nproj = components[0][1].shape[1]
                block_R = np.zeros((nproj, nvir_k))
                block_I = np.zeros((nproj, nvir_k))
                for orb_idx, weight in components:
                    block_R += lib.dot(weight.T, self._osv_cache_R[orb_idx][kj])
                    block_I += lib.dot(weight.T, self._osv_cache_I[orb_idx][kj])
                pkR[offset : offset + nproj] = block_R
                pkI[offset : offset + nproj] = block_I
                offset += nproj
            pk_R[kj] = pkR
            pk_I[kj] = pkI

        resultR, resultI = self._contract_proj_blk(
            j_ref, R_idx, pk_R, pk_I, total_nproj
        )

        results = []
        offset = 0
        for nproj in nproj_list:
            results.append(
                (resultR[offset : offset + nproj], resultI[offset : offset + nproj])
            )
            offset += nproj
        return results

    def _build_osv_cache(self, osv_u):
        """Build cached OSV projections for each orbital and k-point."""
        nkpts = len(self.kpts)
        nlo = len(osv_u)
        self._osv_cache_R = []
        self._osv_cache_I = []
        for j in range(nlo):
            u_j = osv_u[j]
            cR = [None] * nkpts
            cI = [None] * nkpts
            for kj in range(nkpts):
                cR[kj] = lib.dot(u_j.conj().T, self._U_k[kj].real.T)
                cI[kj] = lib.dot(u_j.conj().T, self._U_k[kj].imag.T)
            self._osv_cache_R.append(cR)
            self._osv_cache_I.append(cI)

    def build_osv_proj_cache(self, osv_u):
        self._build_osv_cache(osv_u)

    def _project_osv_blk(self, occ_idx, components):
        """Contract cached OSV projections against the occupied block."""
        npc = self._nlo_per_cell
        nkpts = len(self.kpts)
        j_ref = occ_idx % npc
        R_idx = occ_idx // npc
        nproj = components[0][1].shape[1]
        nvir_k = self._osv_cache_R[0][0].shape[1]

        pk_R = [None] * nkpts
        pk_I = [None] * nkpts
        for kj in range(nkpts):
            pkR = np.zeros((nproj, nvir_k))
            pkI = np.zeros_like(pkR)
            for orb_idx, weight in components:
                pkR += lib.dot(weight.T, self._osv_cache_R[orb_idx][kj])
                pkI += lib.dot(weight.T, self._osv_cache_I[orb_idx][kj])
            pk_R[kj] = pkR
            pk_I[kj] = pkI

        return self._contract_proj_blk(j_ref, R_idx, pk_R, pk_I, nproj)

    def _project_cached_osv_blk(self, occ_idx, orb_idx):
        """Return the projected block for a cached OSV basis."""
        if not hasattr(self, '_osv_proj_result_cache'):
            self._osv_proj_result_cache = {}

        key = (occ_idx, orb_idx)
        if key in self._osv_proj_result_cache:
            return self._osv_proj_result_cache[key]

        npc = self._nlo_per_cell
        j_ref = occ_idx % npc
        R_idx = occ_idx // npc
        nproj = self._osv_cache_R[orb_idx][0].shape[0]

        result = self._contract_proj_blk(
            j_ref, R_idx, self._osv_cache_R[orb_idx], self._osv_cache_I[orb_idx], nproj
        )
        self._osv_proj_result_cache[key] = result
        return result

    def _project_cached_osv_blk_batch(self, occ_idx_list, orb_idx_list):
        results = {}
        occ_groups = {}
        for occ_idx, orb_idx in zip(occ_idx_list, orb_idx_list):
            occ_groups.setdefault(occ_idx, []).append(orb_idx)

        for occ_idx, orb_list in occ_groups.items():
            npc = self._nlo_per_cell
            nkpts = len(self.kpts)
            j_ref = occ_idx % npc
            R_idx = occ_idx // npc
            nproj_list = [
                self._osv_cache_R[orb_idx][0].shape[0] for orb_idx in orb_list
            ]
            total_nproj = sum(nproj_list)

            pk_R = [
                np.empty((total_nproj, self._osv_cache_R[orb_list[0]][0].shape[1]))
                for _ in range(nkpts)
            ]
            pk_I = [
                np.empty((total_nproj, self._osv_cache_I[orb_list[0]][0].shape[1]))
                for _ in range(nkpts)
            ]
            offset = 0
            for orb_idx, nproj in zip(orb_list, nproj_list):
                for kj in range(nkpts):
                    pk_R[kj][offset : offset + nproj] = self._osv_cache_R[orb_idx][kj]
                    pk_I[kj][offset : offset + nproj] = self._osv_cache_I[orb_idx][kj]
                offset += nproj

            resultR, resultI = self._contract_proj_blk(
                j_ref, R_idx, pk_R, pk_I, total_nproj
            )
            offset = 0
            for orb_idx, nproj in zip(orb_list, nproj_list):
                results[(occ_idx, orb_idx)] = (
                    resultR[offset : offset + nproj],
                    resultI[offset : offset + nproj],
                )
                offset += nproj

        return results

    def _build_kspace_data(self, log, cput1):
        """Build the compact k-space data used by periodic ERI paths."""
        npc = self._nlo_per_cell
        nkpts = len(self.kpts)
        nocc = self.nocc
        naux_by_q = self.naux_by_q
        naux = self.naux
        Naux = self.Naux_ibz
        nqpts = len(self.qpts_ibz)
        nao = self.cell.nao_nr()
        nvir = self.nvir
        COMPLEX = np.complex128

        kscf = self._kscf
        nocc_k = kscf.cell.nelectron // 2
        nmo_k = kscf.mo_coeff[0].shape[1]
        frozen_k = nocc_k - npc
        vir_idx = list(range(0, frozen_k)) + list(range(nocc_k, nmo_k))
        nvir_k = len(vir_idx)

        k_can_vir = [
            np.asarray(kscf.mo_coeff[ki][:, vir_idx], order='C') for ki in range(nkpts)
        ]
        k_vir_sc = self.s2k_mo_coeff(self.vir_coeff)

        s_k = kscf.get_ovlp()
        self._U_k = []
        for kj in range(nkpts):
            U = lib.dot(k_can_vir[kj].conj().T, lib.dot(s_k[kj], k_vir_sc[kj]))
            self._U_k.append(U)

        if log.verbose >= logger.DEBUG:
            for kj in range(nkpts):
                U = self._U_k[kj]
                recon = lib.dot(k_can_vir[kj], U)
                err = np.linalg.norm(recon - k_vir_sc[kj])
                occ_idx = list(range(frozen_k, nocc_k))
                k_occ_active = kscf.mo_coeff[kj][:, occ_idx]
                overlap = lib.dot(k_occ_active.conj().T, lib.dot(s_k[kj], k_vir_sc[kj]))
                log.debug(
                    'proj check kj=%d: ||recon_err||=%.2e  ||occ_overlap||=%.2e  '
                    'nmo_k=%d nao=%d',
                    kj,
                    err,
                    np.linalg.norm(overlap),
                    nmo_k,
                    nao,
                )

        ref_occ = self.occ_coeff[:, :npc]
        k_occ_ref = self.s2k_mo_coeff(ref_occ)
        self._phase_factor = np.sqrt(nkpts) * self.phase

        if log.verbose >= logger.DEBUG:
            k_occ_full = self.s2k_mo_coeff(self.occ_coeff)
            sqN = np.sqrt(nkpts)
            max_ts_err = 0.0
            for T in range(1, min(nkpts, 5)):
                for j_ref in range(npc):
                    err2 = sum(
                        np.linalg.norm(
                            sqN * self.phase[T, ki].conj() * k_occ_ref[ki][:, j_ref]
                            - k_occ_full[ki][:, T * npc + j_ref]
                        )
                        ** 2
                        for ki in range(nkpts)
                    )
                    norm2 = sum(
                        np.linalg.norm(k_occ_full[ki][:, T * npc + j_ref]) ** 2
                        for ki in range(nkpts)
                    )
                    max_ts_err = max(max_ts_err, np.sqrt(err2 / max(norm2, 1e-30)))
            if max_ts_err > 1e-6:
                log.warn(
                    'Occupied orbitals lack translational symmetry '
                    '(max rel err %.2e). Use sort_orb_by_cell_ts() '
                    'for TS-preserving orbital sorting.',
                    max_ts_err,
                )
            k_occ_full = None

        self._qi_ranges = []
        self._kj_for_ki_qi = [dict() for _ in range(nkpts)]
        for qi, q in enumerate(self.ibz2bz):
            nauxq = naux_by_q[q]
            b0_q = naux * qi
            self._qi_ranges.append((b0_q, nauxq))
            for ki, kj in self.kikj_by_q[q]:
                assert qi not in self._kj_for_ki_qi[ki], (
                    f'Duplicate ki={ki} for qi={qi}: kj={kj} vs {self._kj_for_ki_qi[ki][qi]}'
                )
                self._kj_for_ki_qi[ki][qi] = kj

        mem_df = 2 * nkpts * npc * nvir_k * Naux * 8 / 1024**2.0
        mem_full = 2 * nocc * nvir * Naux * 8 / 1024**2.0
        mem_avail = self.max_memory - lib.current_memory()[0]
        log.info(
            'ao2mo kspace data: est mem= %.2f MB  avail mem= %.2f MB  '
            '(nkpts=%d, npc=%d, nvir_k=%d, Naux=%d, full= %.2f MB)',
            mem_df,
            mem_avail,
            nkpts,
            npc,
            nvir_k,
            Naux,
            mem_full,
        )

        self._ovL_k_R = np.zeros((nkpts, npc, nvir_k, Naux))
        self._ovL_k_I = np.zeros((nkpts, npc, nvir_k, Naux))

        tao = []
        ao_loc = None

        for qi, q in enumerate(self.ibz2bz):
            nauxq = naux_by_q[q]
            w = self.qpts_ibz_weights[qi]
            b0_q = naux * qi
            for (ki, kj), Lpq_ao in self.loop(q):
                if Lpq_ao[0].size != nao**2:
                    Lpq_ao = lib.unpack_tril(Lpq_ao).astype(COMPLEX)
                mo = np.asarray(np.hstack((k_occ_ref[ki], k_can_vir[kj])), order='F')
                ijslice = (0, npc, npc, npc + nvir_k)
                Lpq = _ao2mo.r_e2(Lpq_ao, mo, ijslice, tao, ao_loc)
                Lpq = Lpq.reshape(nauxq, npc, nvir_k)
                LpqR = (w * Lpq.real).transpose(1, 2, 0)  # (npc, nvir_k, nauxq)
                LpqI = (w * Lpq.imag).transpose(1, 2, 0)
                Lpq_ao = Lpq = None

                self._ovL_k_R[ki][:, :, b0_q : b0_q + nauxq] += LpqR
                self._ovL_k_I[ki][:, :, b0_q : b0_q + nauxq] += LpqI
                LpqR = LpqI = None
            cput1 = log.timer('kspace ao2mo for qidx %d/%d' % (qi + 1, nqpts), *cput1)

        if log.verbose >= logger.DEBUG:
            qi0, q0 = 0, self.ibz2bz[0]
            nauxq0 = naux_by_q[q0]
            w0 = self.qpts_ibz_weights[qi0]
            b0_0 = naux * qi0
            for (ki0, kj0), Lpq_ao_d in self.loop(q0):
                if Lpq_ao_d[0].size != nao**2:
                    Lpq_ao_d = lib.unpack_tril(Lpq_ao_d).astype(COMPLEX)
                mo_d = np.asarray(
                    np.hstack((k_occ_ref[ki0], k_can_vir[kj0])), order='F'
                )
                Lpq_d = _ao2mo.r_e2(
                    Lpq_ao_d, mo_d, (0, npc, npc, npc + nvir_k), [], None
                )
                Lpq_d = Lpq_d.reshape(nauxq0, npc, nvir_k)
                ref_R = (w0 * Lpq_d.real).transpose(1, 2, 0)
                ref_I = (w0 * Lpq_d.imag).transpose(1, 2, 0)
                stored_R = self._ovL_k_R[ki0][:, :, b0_0 : b0_0 + nauxq0]
                stored_I = self._ovL_k_I[ki0][:, :, b0_0 : b0_0 + nauxq0]
                log.debug(
                    'block check (qi=%d, ki=%d, kj=%d): max|dR|=%.2e max|dI|=%.2e',
                    qi0,
                    ki0,
                    kj0,
                    np.max(np.abs(stored_R - ref_R)),
                    np.max(np.abs(stored_I - ref_I)),
                )
                break

        return cput1

    def _build_incore(self, log, cput1):
        """Build full supercell ``ovLR`` and ``ovLI`` in memory."""
        cput1 = self._build_kspace_data(log, cput1)

        npc = self._nlo_per_cell
        nkpts = len(self.kpts)
        nocc = self.nocc
        nvir = self.nvir
        Naux = self.Naux_ibz
        nqpts = len(self._qi_ranges)

        mem_df = 2 * nocc * nvir * Naux * 8 / 1024**2.0
        mem_avail = self.max_memory - lib.current_memory()[0]
        log.info(
            'incore reconstruct: full ovL %.2f MB, avail %.2f MB', mem_df, mem_avail
        )

        self.ovLR = np.zeros((nocc, nvir, Naux))
        self.ovLI = np.zeros((nocc, nvir, Naux))
        pf = self._phase_factor  # (nkpts, nkpts) complex
        pfR_all = pf.real
        pfI_all = pf.imag

        for qi in range(nqpts):
            b0, nauxq = self._qi_ranges[qi]
            for j_ref in range(npc):
                tR = np.empty((nkpts, nvir, nauxq))
                tI = np.empty((nkpts, nvir, nauxq))
                for ki in range(nkpts):
                    kj = self._kj_for_ki_qi[ki][qi]
                    ovkR = self._ovL_k_R[ki][j_ref, :, b0 : b0 + nauxq]
                    ovkI = self._ovL_k_I[ki][j_ref, :, b0 : b0 + nauxq]
                    U_kj = self._U_k[kj]
                    UR = U_kj.real.T
                    UI = U_kj.imag.T
                    tR[ki] = lib.dot(UR, ovkR) - lib.dot(UI, ovkI)
                    tI[ki] = lib.dot(UI, ovkR) + lib.dot(UR, ovkI)

                tR_2d = tR.reshape(nkpts, nvir * nauxq)
                tI_2d = tI.reshape(nkpts, nvir * nauxq)
                blkR = (lib.dot(pfR_all, tR_2d) - lib.dot(pfI_all, tI_2d)).reshape(
                    nkpts, nvir, nauxq
                )
                blkI = (lib.dot(pfR_all, tI_2d) + lib.dot(pfI_all, tR_2d)).reshape(
                    nkpts, nvir, nauxq
                )
                self.ovLR[j_ref::npc, :, b0 : b0 + nauxq] = blkR
                self.ovLI[j_ref::npc, :, b0 : b0 + nauxq] = blkI
            cput1 = log.timer('incore reconstruct qidx %d/%d' % (qi + 1, nqpts), *cput1)

        self._ovL_k_R = None
        self._ovL_k_I = None
        self._U_k = None
        self._phase_factor = None
        self._qi_ranges = None
        self._kj_for_ki_qi = None

    def _build_outcore(self, log, cput1):
        """Build full supercell ``ovLR`` and ``ovLI`` on disk."""
        cput1 = self._build_kspace_data(log, cput1)

        npc = self._nlo_per_cell
        nkpts = len(self.kpts)
        nocc = self.nocc
        nvir = self.nvir
        Naux = self.Naux_ibz
        nqpts = len(self._qi_ranges)

        disk_gb = 2 * nocc * nvir * Naux * 8 / 1e9
        log.info('outcore reconstruct: %.2f GB to disk', disk_gb)

        self._h5file = lib.H5TmpFile()
        log.info('outcore file: %s', self._h5file.filename)
        chunks = (1, nvir, min(Naux, 8192))
        self.ovLR = self._h5file.create_dataset(
            'ovLR', (nocc, nvir, Naux), dtype='f8', chunks=chunks
        )
        self.ovLI = self._h5file.create_dataset(
            'ovLI', (nocc, nvir, Naux), dtype='f8', chunks=chunks
        )

        pf = self._phase_factor
        pfR_all = pf.real
        pfI_all = pf.imag

        for qi in range(nqpts):
            b0, nauxq = self._qi_ranges[qi]

            bufR = np.empty((nkpts, npc, nvir, nauxq))
            bufI = np.empty((nkpts, npc, nvir, nauxq))
            for j_ref in range(npc):
                tR = np.empty((nkpts, nvir, nauxq))
                tI = np.empty((nkpts, nvir, nauxq))
                for ki in range(nkpts):
                    kj = self._kj_for_ki_qi[ki][qi]
                    ovkR = self._ovL_k_R[ki][j_ref, :, b0 : b0 + nauxq]
                    ovkI = self._ovL_k_I[ki][j_ref, :, b0 : b0 + nauxq]
                    U_kj = self._U_k[kj]
                    UR = U_kj.real.T
                    UI = U_kj.imag.T
                    tR[ki] = lib.dot(UR, ovkR) - lib.dot(UI, ovkI)
                    tI[ki] = lib.dot(UI, ovkR) + lib.dot(UR, ovkI)
                tR_2d = tR.reshape(nkpts, nvir * nauxq)
                tI_2d = tI.reshape(nkpts, nvir * nauxq)
                bufR[:, j_ref] = (
                    lib.dot(pfR_all, tR_2d) - lib.dot(pfI_all, tI_2d)
                ).reshape(nkpts, nvir, nauxq)
                bufI[:, j_ref] = (
                    lib.dot(pfR_all, tI_2d) + lib.dot(pfI_all, tR_2d)
                ).reshape(nkpts, nvir, nauxq)

            for T in range(nkpts):
                self.ovLR[T * npc : (T + 1) * npc, :, b0 : b0 + nauxq] = bufR[T]
                self.ovLI[T * npc : (T + 1) * npc, :, b0 : b0 + nauxq] = bufI[T]
            cput1 = log.timer(
                'outcore reconstruct qidx %d/%d' % (qi + 1, nqpts), *cput1
            )

        self._refcell_ovLR = np.asarray(self.ovLR[:npc], order='C')
        self._refcell_ovLI = np.asarray(self.ovLI[:npc], order='C')

        self._ovL_k_R = None
        self._ovL_k_I = None
        self._U_k = None
        self._phase_factor = None
        self._qi_ranges = None
        self._kj_for_ki_qi = None
