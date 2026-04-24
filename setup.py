#!/usr/bin/env python

"""The setup script."""

from setuptools import setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

setup(
    author="Saket Choudhary",
    author_email="saketkc@gmail.com",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    description="Python implementation of SCTransform for single-cell" +
                "RNA-seq data normalization",
    entry_points={
        "console_scripts":
            ["pysctransform=pysctransform.cli:parse_args"]
    },
    license="BSD license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="pysctransform",
    name="pysctransform",
    packages=["pysctransform"],
    python_requires=">=3.10",
    test_suite="tests",
    url="https://github.com/saketkc/pysctransform",
    version="0.1.1",
    zip_safe=False,
)
