#!/usr/bin/env python3

from setuptools import setup

with open("README.md","r") as fh:
    long_description = fh.read()

setup(
name='lamp', #pypi name
version = '0.0.10',
description='Python Code Base Used by the Holobiome Team',
py_modules=['lamp',
            ], #'import' name
package_dir={'':'src'},
long_description=long_description,
long_description_content_type="text/markdown",
install_requires=["pandas"],
extras_require={"dev":["pytest>=3.7",]},
url = "https://github.com/HolobiomeProject/Holotools", #sdist
author="Barry Guglielmo",
author_email="barryguglielmo@gmail.com"
)
