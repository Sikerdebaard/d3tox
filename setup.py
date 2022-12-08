from setuptools import setup, find_packages
from setuptools.command.egg_info import egg_info

import sys


class EggInfoEx(egg_info):
    """Includes license file into `.egg-info` folder."""

    def run(self):
        # don't duplicate license into `.egg-info` when building a distribution
        if not self.distribution.have_run.get('install', True):
            # `install` command is in progress, copy license
            self.mkpath(self.egg_info)
            self.copy_file('LICENSE', self.egg_info)

        egg_info.run(self)


if sys.version_info < (3, 6):
    sys.exit('Sorry, Python < 3.6 is not supported')

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='d3tox',
    version='1',
    license_files = ('LICENSE', ),
    cmdclass = {'egg_info': EggInfoEx},
    description='Degenerative Disc Disease in the neck ToolbOX',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Thomas Phil',
    author_email='thomas@tphil.nl',
    url='https://github.com/Sikerdebaard/d3tox',
    python_requires=">=3.6",
    packages=find_packages(),  # same as name
    install_requires=[
        'tqdm>=4.64.0',
        'cleo>=0.8.1',
        'pandas>=1.0.3',
        'nibabel>=4.0.2',
        'scipy>=1.9.1',
        'scikit-image>=0.19.3',
        'numpy>=1.23.3',
        'joblib>=1.1.0',
        'tqdm>=4.64.1',
        'psutil>=5.9.2',
    ],
    entry_points={
        'console_scripts': [
            'd3tox=d3tox.cli:run',
        ],
    },
)