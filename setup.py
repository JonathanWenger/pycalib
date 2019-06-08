# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pycalib',
    version='0.1.0',
    description='Non-parametric Calibration for Classification',
    long_description=readme,
    author='Jonathan Wenger',
    author_email='j.wenger@tum.de',
    url='https://github.com/JonathanWenger/pycalib',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'data', 'figures'))
)
