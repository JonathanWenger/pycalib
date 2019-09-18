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
    author_email='jonathan.wenger@uni-tuebingen.de',
    url='https://github.com/JonathanWenger/pycalib',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'data', 'figures')),
    install_requires=
    [
        'betacal>=0.2.7',
        'gpflow>=1.3.0',
        'matplotlib>=3.1.0',
        'numpy>=1.15.4',
        'pandas>=0.23.4',
        'scikit-learn>=0.20.1',
        'scipy>=1.3.0',
        'tensorflow>=1.12.2',
        'torch>=0.4.1',
        'pretrainedmodels>=0.7.4',
        'scikit-garden @ git+ssh://git@github.com/scikit-garden/scikit-garden@master#egg=scikit-garden'
    ]
)

# Installation dependency from github:
# https://stackoverflow.com/questions/32688688/how-to-write-setup-py-to-include-a-git-repo-as-a-dependency