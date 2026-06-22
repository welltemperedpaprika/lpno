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

"""PAO-basis helpers shared by periodic kdOSV and KPNO drivers."""

import numpy as np
from pyscf import lib
from pyscf.lib import logger

from pyscf.pbc.lpno import kpao as kpao_mod
from pyscf.pbc.lpno import kpair_domain as pd_mod
from pyscf.pbc.lpno import kpao_pair_domain as pao_pd_mod
from pyscf.lpno.tools import summarize_domain


PAO_KEYS = (
    "pbc_pao_domain_mode",
    "pbc_pao_bp_thresh",
    "pbc_pao_principal_thresh",
    "pbc_pao_q_thresh",
    "pbc_pao_pair_screening",
)


def set_default_options(mp):
    mp.pbc_pao_domain_mode = "full"
    mp.pbc_pao_bp_thresh = 0.999
    mp.pbc_pao_principal_thresh = 0.99
    mp.pbc_pao_q_thresh = None
    mp.pbc_pao_pair_screening = False
    mp._pao = None


def dump_flags(mp, log):
    log.info("pbc_pao_domain_mode = %s", mp.pbc_pao_domain_mode)
    if mp.pbc_pao_domain_mode == "bp":
        log.info("pbc_pao_bp_thresh = %.6f", mp.pbc_pao_bp_thresh)
    if mp.pbc_pao_domain_mode in ("principal", "doi"):
        log.info("pbc_pao_principal_thresh = %.6f",
                 mp.pbc_pao_principal_thresh)
    log.info("pbc_pao_pair_screening = %s", mp.pbc_pao_pair_screening)
    if mp.pbc_pao_pair_screening:
        log.info("dipole_mode = %s", getattr(mp, "dipole_mode", "ewald"))


def make_pao_full(mp):
    vir_coeff = mp.split_mo_coeff()[2]
    vir_energy = mp.split_mo_energy()[2]
    domain_mode = getattr(mp, "pbc_pao_domain_mode", "full")
    ref_atom_domains = None
    if domain_mode != "full":
        ref_atom_domains = kpao_mod.reference_atom_domains(
            mp.cell,
            mp.lo_coeff,
            mp.nlo_per_cell,
            mode=domain_mode,
            s1e=mp.s1e,
            bp_thr=getattr(mp, "pbc_pao_bp_thresh", 0.999),
            q_thr=getattr(mp, "pbc_pao_q_thresh", None),
            principal_thr=getattr(mp, "pbc_pao_principal_thresh", 0.99),
        )
    return kpao_mod.PeriodicPAO(
        mol=mp.cell,
        occ_coeff=mp.lo_coeff,
        nlo_per_cell=mp.nlo_per_cell,
        nlo=mp.nlo,
        cell_sub=mp._cell_sub,
        vir_coeff=vir_coeff,
        vir_energy=vir_energy,
        ref_atom_domains=ref_atom_domains,
        s1e=mp.s1e,
    )


def make_osv(mp, eris, moe_lo, thresh_osv=None):
    log = logger.new_logger(mp)
    pao = mp.make_pao_full()
    mp._pao = pao
    if hasattr(pao, "domain_summary"):
        summarize_domain(
            pao.domain_summary()["ref_domain_sizes"],
            log,
            "PAO orbital domain size",
        )
    if thresh_osv is None:
        osv_param = mp._get_osv_param()
    else:
        osv_param = {"thresh": thresh_osv}
    osv_param.update({
        "nlo_per_cell": mp.nlo_per_cell,
        "nlo": mp.nlo,
    })
    osv = kpao_mod.PeriodicPAOBasisOSV(
        eris, moe_lo, pao, osv_param, verbose=mp.verbose)
    summarize_domain(osv.nosv, log, "PAO-basis OSV domain size")
    if mp.chkfile:
        lib.chkfile.dump(mp.chkfile, "osv/nosv", osv.nosv)
    return osv


def full_pair_mask(mp):
    pair_mask = np.ones((mp.nlo_per_cell, mp.nlo), dtype=bool)
    return mp._get_pair_index().validate_ref_pair_mask(
        pair_mask, label="PAO pair", require_diagonal=True)


def zero_pair_energy(mp):
    zero = np.zeros((mp.nlo_per_cell, mp.nlo), dtype=float)
    return lib.tag_array(
        zero, e_corr_ss=zero.copy(), e_corr_os=zero.copy())


def make_pair_screen(mp, osv, log):
    """Return PAO near-pair mask and distant-pair energy estimate."""
    if not getattr(mp, "pbc_pao_pair_screening", False):
        pair_mask = full_pair_mask(mp)
        return pair_mask, zero_pair_energy(mp)

    moe_lo = np.diag(mp.foo)
    dipole_mode = getattr(mp, "dipole_mode", "ewald")
    if dipole_mode == "ewald":
        epair, Rpair = pd_mod.pbc_pair_ewald_dipole_pao(
            mp.cell, mp.lo_coeff, mp.nlo_per_cell, moe_lo, mp._pao, osv)
    elif dipole_mode == "molecular":
        epair, Rpair = pd_mod.pbc_pair_dipole_pao(
            mp.cell, mp.lo_coeff, mp.nlo_per_cell, moe_lo, mp._pao, osv)
    else:
        raise NotImplementedError(
            f"PAO pair screening does not support dipole_mode={dipole_mode!r}")

    if getattr(mp, "screen_mode", "energy") == "energy":
        pair_mask = np.abs(epair) > mp.thresh_distpair
        for i in range(mp.nlo_per_cell):
            pair_mask[i, i] = True
        log.info(
            "PAO energy-pair prescreen (%s, thresh=%.1e): %d Near | %d Dist | %d Total pairs",
            dipole_mode,
            mp.thresh_distpair,
            np.count_nonzero(pair_mask),
            np.count_nonzero(~pair_mask),
            pair_mask.size,
        )
    else:
        pair_mask = Rpair <= mp.rmin_distpair
        for i in range(mp.nlo_per_cell):
            pair_mask[i, i] = True
        log.info(
            "PAO dist-pair prescreen (%s, rmin=%.1f): %d Near | %d Dist | %d Total pairs",
            dipole_mode,
            mp.rmin_distpair,
            np.count_nonzero(pair_mask),
            np.count_nonzero(~pair_mask),
            pair_mask.size,
        )

    pair_mask = mp._get_pair_index().validate_ref_pair_mask(
        pair_mask, label="PAO pair", require_diagonal=True)
    dist_pair_mask = ~pair_mask
    epair_dist = epair * dist_pair_mask
    epair_dist = lib.tag_array(
        epair_dist,
        e_corr_ss=epair_dist * 0.5,
        e_corr_os=epair_dist * 0.5,
    )
    if np.any(dist_pair_mask):
        log.info(
            "PAO dist-pair energy estimate (dipole approx): %.9f Ha",
            np.sum(epair_dist),
        )

    return pair_mask, epair_dist


def make_pair_work_plan(mp, pair_mask, epair_dist, log,
                        include_residual_coupling=False):
    """Build the PAO pair masks used by prescreen, domains, and residuals."""
    pair_mask = pair_mask.copy()
    if (getattr(mp, "use_doi_screening", False)
            and hasattr(mp, "_apply_doi_pair_promotion")):
        pair_mask, epair_dist, doi_promoted = mp._apply_doi_pair_promotion(
            pair_mask, epair_dist)
        log.info(
            "PAO DOI-near promotion (%s > %.1e): %d Promoted | %d Near | %d Dist | %d Total pairs",
            mp.doi_metric,
            mp.thresh_doi,
            np.count_nonzero(doi_promoted),
            np.count_nonzero(pair_mask),
            np.count_nonzero(~pair_mask),
            pair_mask.size,
        )

    use_scmp2_coupling = (
        include_residual_coupling
        and getattr(mp, "use_scmp2_coupling", False)
        and not getattr(mp, "use_doi_screening", False)
        and np.any(~pair_mask)
        and hasattr(mp, "_residual_coupling_mask")
    )
    pair_mask_residual = (
        mp._residual_coupling_mask(pair_mask)
        if use_scmp2_coupling else pair_mask
    )
    if use_scmp2_coupling:
        log.info(
            "PAO residual-coupling pair domain: %d Energy-near | %d Coupling-only | %d Total pairs",
            np.count_nonzero(pair_mask),
            np.count_nonzero(pair_mask_residual & ~pair_mask),
            np.count_nonzero(pair_mask_residual),
        )

    if hasattr(mp, "_expand_pair_mask"):
        pair_mask_full = mp._expand_pair_mask(pair_mask_residual)
    else:
        pair_mask_full = mp._get_pair_index().expand_ref_pair_mask(
            pair_mask_residual)

    return {
        "pair_mask": pair_mask,
        "pair_mask_residual": pair_mask_residual,
        "pair_mask_full": pair_mask_full,
        "near_list": list(zip(*np.where(pair_mask))),
        "domain_list": list(zip(*np.where(pair_mask_residual))),
        "epair_dist": epair_dist,
        "use_scmp2_coupling": use_scmp2_coupling,
    }


def record_pair_work_summary(
    mp, pair_mask, pair_mask_residual, pair_mask_strong, pair_mask_full,
    fock_pair_blocks=0,
):
    summary = {
        "pao_pair_screening_enabled": bool(
            getattr(mp, "pbc_pao_pair_screening", False)),
        "near_pairs": int(np.count_nonzero(pair_mask)),
        "dist_pairs": int(pair_mask.size - np.count_nonzero(pair_mask)),
        "residual_pairs": int(np.count_nonzero(pair_mask_residual)),
        "strong_pairs": int(np.count_nonzero(pair_mask_strong)),
        "overlap_pair_blocks": int(np.count_nonzero(pair_mask_full)),
        "fock_pair_blocks": int(fock_pair_blocks),
    }
    mp._pair_work_summary = summary


def make_kdosv_pair_domain(mp, eris, osv, timer, cput2, log):
    pair_mask, epair_dist = make_pair_screen(mp, osv, log)
    pair_work = make_pair_work_plan(mp, pair_mask, epair_dist, log)
    pair_mask = pair_work["pair_mask"]
    pair_mask_full = pair_work["pair_mask_full"]
    summarize_domain(
        mp._pao.pair_domain_sizes(pair_mask),
        log,
        "PAO pair domain size",
    )
    mp._update_pair_mask("Near pair", pair_mask)
    mp._update_pair_mask("Residual pair", pair_work["pair_mask_residual"])
    mp._update_pair_mask("Dist pair", ~pair_mask)
    mp._update_pair_energy("Dist pair", pair_work["epair_dist"], ~pair_mask)
    cput3 = log.timer("Pair prescreen", *cput2)
    timer["Pair prescreen"] = np.asarray(cput3) - np.asarray(cput2)

    osv.get_ovlp(pair_mask=pair_mask_full)

    pair_domain = pao_pd_mod.PairDomain_kdOSV_PAOBasis(
        eris, osv, mp._pao, pair_mask, mp.nlo_per_cell, mp.nlo,
        pair_work["near_list"],
        cell_sub=mp._cell_sub)
    mp._update_pair_mask("Strong pair", pair_domain.pair_mask)
    record_pair_work_summary(
        mp, pair_mask, pair_work["pair_mask_residual"],
        pair_domain.pair_mask, pair_mask_full)

    mp._clear_eris(eris)

    cput4 = log.timer("Pair domain (PAO-basis JK build)", *cput3)
    timer["Pair domain"] = np.asarray(cput4) - np.asarray(cput3)
    return pair_domain


def make_kpno_pair_domain(mp, eris, osv, timer, cput2, log):
    pair_mask, epair_dist = make_pair_screen(mp, osv, log)
    pair_work = make_pair_work_plan(
        mp, pair_mask, epair_dist, log, include_residual_coupling=True)
    pair_mask = pair_work["pair_mask"]
    pair_mask_residual = pair_work["pair_mask_residual"]
    pair_mask_full = pair_work["pair_mask_full"]
    summarize_domain(
        mp._pao.pair_domain_sizes(pair_mask),
        log,
        "PAO pair domain size",
    )
    mp._update_pair_mask("Near pair", pair_mask)
    mp._update_pair_mask("Residual pair", pair_mask_residual)
    mp._update_pair_mask("Dist pair", ~pair_mask)
    mp._update_pair_energy("Dist pair", pair_work["epair_dist"], ~pair_mask)

    cput3 = log.timer("Pair prescreen", *cput2)
    timer["Pair prescreen"] = np.asarray(cput3) - np.asarray(cput2)

    osv.get_ovlp(pair_mask=pair_mask_full)
    osv.get_fock(pair_mask=pair_mask_full)

    moe_lo = np.diag(mp.foo)
    vir_energy = mp.split_mo_energy()[2]
    pair_domain = pao_pd_mod.PairDomain_kPNO_PAOBasis(
        eris,
        osv,
        mp._pao,
        mp.pno_param,
        moe_lo,
        vir_energy,
        pair_mask=pair_mask_residual,
        nlo_per_cell=mp.nlo_per_cell,
        nlo=mp.nlo,
        thresh_psv_lindep=mp.thresh_psv_lindep,
        thresh_weakpair=mp.thresh_weakpair,
        with_ex_ene=mp.with_ex_ene,
        compress_diagpair=mp.compress_diagpair,
        cell_sub=mp._cell_sub,
        keep_weak_domains=pair_work["use_scmp2_coupling"],
    )

    mp._clear_eris(eris)
    pair_domain.eris = None
    osv.F = None

    pair_domain.pair_mask_residual = pair_mask_residual
    pair_mask_strong = pair_mask & pair_domain.pair_mask_strong
    pair_domain.pair_mask_strong = pair_mask_strong

    mp._update_pair_mask("Strong pair", pair_mask_strong)
    record_pair_work_summary(
        mp,
        pair_mask,
        pair_mask_residual,
        pair_mask_strong,
        pair_mask_full,
        fock_pair_blocks=np.count_nonzero(pair_mask_full),
    )

    pair_mask_weak = pair_mask & (~pair_mask_strong)
    mp._update_pair_mask("Weak pair", pair_mask_weak)

    if hasattr(pair_domain, "epair_psv"):
        mp._update_pair_energy(
            "Weak pair",
            mp._mask_pair_energy(pair_domain.epair_psv, pair_mask_weak))

    if mp.pno_param is not None and hasattr(pair_domain, "epair_pno"):
        mp._update_pair_energy(
            "PNO truncation",
            mp._mask_pair_energy(pair_domain.epair_pno, pair_mask_strong))

    summarize_domain(pair_domain.npsv, log, "PNO domain size")

    cput4 = log.timer("Pair domain (PAO-basis PNO build)", *cput3)
    timer["Pair domain"] = np.asarray(cput4) - np.asarray(cput3)
    return pair_domain
