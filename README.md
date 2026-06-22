# lpno

local MP2 in pyscf with pbc support

## install

```bash
git clone https://github.com/welltemperedpaprika/lpno.git
cd lpno
pip install -e .
```

## usage

```python
from pyscf.pbc import gto, scf
from pyscf.pbc.lpno import KPNOMP2   # or kdOSVMP2

cell = gto.Cell()
cell.atom = 'He 0 0 0'
cell.a = '3.0 0 0; 0 3.0 0; 0 0 3.0'
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.build()

kmesh = [2, 2, 2]
kpts = cell.make_kpts(kmesh)
kmf = scf.KRHF(cell, kpts).density_fit().run()

lo_coeff = ...   # localized occupied orbitals, see examples

mp = KPNOMP2(kmf, lo_coeff, kmesh=kmesh)
mp.thresh_pno = 1e-8
mp.kernel()
print(mp.e_corr)
```

the methods take localized occupied orbitals as input. the scripts in
`examples/pbc/lpno/` show two ways to get them: one using only stock pyscf
(`00-kpnomp2.py`), and one using periodic Pipek-Mezey (`01-kdosvmp2_kpipek.py`).

## accuracy

KPNO-MP2 converges to canonical periodic MP2 as the PNO threshold is tightened,
and the correlation part of the cohesive energy extrapolates to the TDL along
with canonical k-point MP2.

![silicon PNO convergence](figures/si_pno_accuracy.png)

![ammonia cohesive energy vs k-mesh](figures/ammonia_kmesh.png)

the scripts and data for these are in `figures/`.

## tests

```bash
pytest pyscf/pbc/lpno/test
```

## citation

if you use this, please cite the accompanying paper (Liang, Yang, Ye, and
Berkelbach; in preparation).

## license

Apache 2.0, see LICENSE.
