#!/usr/bin/env python
from pathlib import Path

from setuptools import setup, find_namespace_packages

readme = (Path(__file__).parent / 'README.md').read_text()

setup(
    name='lpno',
    version='0.1.0',
    description='Periodic local MP2 (DLPNO / OSV) for PySCF',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Yu Hsuan Liang, Gengzhi Yang, Hong-Zhou Ye, Timothy C. Berkelbach',
    url='https://github.com/welltemperedpaprika/lpno',
    license='Apache-2.0',
    packages=find_namespace_packages(include=['pyscf.*']),
    python_requires='>=3.9',
    install_requires=['pyscf>=2.14.0', 'numpy', 'scipy', 'h5py'],
    include_package_data=True,
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
)
