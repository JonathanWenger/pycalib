from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pycalib',
    version='0.1.0',
    description='Non-Parametric Calibration for Classification',
    long_description=readme,
    author='Jonathan Wenger',
    author_email='jonathan.wenger@uni-tuebingen.de',
    url='https://github.com/JonathanWenger/pycalib',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'datasets', 'figures', 'benchmark')),
    install_requires=
    [
        'betacal>=0.2.7',
        'gpflow>=1.3.0,<2.0.0',
        'matplotlib>=3.1.0',
        'numpy>=1.15.4',
        'pandas>=0.23.4',
        'scikit-learn>=0.20.1',
        'xgboost>=0.82',
        'scipy>=1.3.0',
        'tensorflow>=1.12.2,<2.0.0',
        'torch>=0.4.1',
        'torchvision>=0.5.0',
        'pretrainedmodels>=0.7.4'
    ]
)

# Installation dependency from github:
# https://stackoverflow.com/questions/32688688/how-to-write-setup-py-to-include-a-git-repo-as-a-dependency