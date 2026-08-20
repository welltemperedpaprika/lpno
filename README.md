# Periodic Local MP2 for PySCF

[![Tests](https://github.com/welltemperedpaprika/lpno/actions/workflows/tests.yml/badge.svg)](https://github.com/welltemperedpaprika/lpno/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Periodic local MP2 (PNO-MP2 and OSV-MP2) with $k$-point sampling and density fitting for [PySCF](https://github.com/pyscf/pyscf).

## Implemented Methods

- **`KPNOMP2`**: Periodic pair natural orbital MP2 with Ewald-screened pair domains.
- **`kdOSVMP2`**: Periodic diagonal orbital-specific virtual MP2.

## Installation

```bash
git clone https://github.com/welltemperedpaprika/lpno.git
cd lpno
pip install -e .
```

Requires PySCF ($\ge$ 2.14.0), NumPy, and SciPy.

## Quick Start

```python
import numpy as np
from pyscf.pbc import gto, scf
from pyscf.pbc.lpno import KPNOMP2, make_lo_kpipek

cell = gto.Cell()
cell.atom = 'Li 0 0 0; H 2.042 2.042 2.042'
cell.a = '''0     2.042 2.042
     2.042 0     2.042
     2.042 2.042 0'''
cell.basis = 'pobtzvp'
cell.build()

kmesh = [2, 2, 2]
kpts = cell.make_kpts(kmesh, time_reversal_symmetry=True)
kmf = scf.KRHF(cell, kpts).density_fit(auxbasis='weigend').run()

# Localize occupied orbitals via periodic Pipek-Mezey (kpipek)
lo_coeff, kfrozen = make_lo_kpipek(cell, kmf, kpts, kmesh)

# Run KPNO-MP2
mp = KPNOMP2(kmf, lo_coeff, frozen=kfrozen, kmesh=kmesh)
mp.thresh_pno = 1e-7
mp.kernel()
print(f"E(corr) = {mp.e_corr:.8f} Ha")
```

For further usage patterns, see `examples/pbc/lpno/`:
- `00-kpnomp2.py`: Setting thresholds (`thresh_pno`, `thresh_osv`, `thresh_weakpair`, `thresh_distpair`).
- `01-kdosvmp2.py`: Periodic OSV-MP2 with dipole prescreening modes (`dipole_mode='ewald'`).
- `02-threshold_ladder.py`: Multi-threshold PNO scans sharing transformed integrals.
- `03-kmp2_comparison.py`: Comparing against canonical `KMP2` and in-core/out-of-core memory modes.

## Tests

```bash
pytest -v
```

## Citation

```bibtex
@article{liang2026periodic,
  title   = {Periodic local MP2 with pair natural orbitals and orbital-specific virtuals},
  author  = {Liang, Yu Hsuan and Yang, Gengzhi and Ye, Hong-Zhou and Berkelbach, Timothy C.},
  journal = {in preparation},
  year    = {2026}
}
```

## Authors

- Yu Hsuan Liang
- Gengzhi Yang
- Hong-Zhou Ye
- Timothy C. Berkelbach

See [`AUTHORS`](AUTHORS) for details.

## License

Apache 2.0 (see [LICENSE](LICENSE)).
