# LPNO-MP2 Benchmark Results

Consolidated dataset of periodic LPNO-MP2 (`KPNOMP2`) calculations across validated production runs (system $\times$ basis $\times$ $k$-mesh $\times$ threshold tuple; 360 rows across 13 chemical systems). Canonical KMP2 reference values and relative percent errors are included on each row where a matched reference exists. Additional calculation details (energy decomposition, per-stage wall times, domain size ranges) are available in the project registry upon request.

---

## Columns

- **`system`**: Chemical system (`ammonia`, `benzene`, `bn`, `c`, `co2`, `ice`, `lih`, `mgo`, `mos2`, `oxaca`, `oxacb`, `si`, `urea`).
- **`basis`**: AO basis set (e.g. `gth-dzvp`, `gth-tzvp`, `cc-pvdz`, `cc-pvtz`, `pob-tzvp`).
- **`kmesh`** / **`nk`**: $k$-point mesh dimensions and total number of $k$-points.
- **Thresholds**:
  - `tp`: PNO occupation cutoff.
  - `tw`: Weak-pair energy cutoff.
  - `td`: Distant-pair energy cutoff (Ewald-dipole estimate).
  - `t_osv`: OSV cutoff.
- **Pair Tiers & Retained Spaces**:
  - `nocc` / `nmo`: Occupied and total molecular orbital counts in the supercell.
  - `npairs_strong`: Strong pairs (iterated and PNO-compressed). Blank for older runs where strong/weak breakdown was not logged.
  - `npairs_weak`: Weak pairs (semicanonical MP2).
  - `npairs_dist`: Distant pairs (dipole-screened).
  - `npairs_total`: Total pairs ($N_{\text{occ}}^{\text{cell}} \times N_{\text{occ}}^{\text{supercell}}$).
  - `osv_avg` / `pno_avg`: Average retained virtuals per orbital (OSV) and per pair (PNO) after truncation.
- **Energies & Errors**:
  - `e_corr`: Correlation energy per unit cell (Hartree).
  - `e_kmp2_ref`: Matched canonical KMP2 reference correlation energy.
  - `err_vs_kmp2_pct`: Relative correlation energy error vs canonical reference (%).
- **Timings & Provenance**:
  - `wall_total`: Total calculation wall time in seconds (from the newest run of this calculation on `node`).
  - `node`: Compute node identifier (compare timings only within one node class).
  - `run_date`: Execution date.

---

## Comparing Wall Times Across Code Eras

The code went through one major performance rewrite; `run_date` identifies the timing era:

- **Before 2026-07-26**: Pre-rewrite code. Wall times are not comparable with later dates (the residual solver and integral transforms were rebuilt that week). Correlation energies are unaffected.
- **2026-07-27 to 2026-08-18**: Rewritten production code; represents the bulk of timed benchmark runs. Comparable within matched node classes.
- **2026-08-19 onward**: Current production code, verified performance-neutral against the previous era on matched nodes.
