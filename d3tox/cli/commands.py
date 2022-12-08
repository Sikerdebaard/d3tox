#!/usr/bin/env python
import psutil

import d3tox.patches  # import monkey patches
from cleo import Command, Application

import logging
import pkg_resources  # part of setuptools

from pathlib import Path

from d3tox.pipeline.preprocessing_pipeline import run_pipeline


class PreprocessPipeline(Command):
    """
    Preprocessing pipeline. Prepares the dataset for classification.

    preprocess
        {--indir= : Path to the data directory.}
        {--outdir= : Path to the output directory. Will be created if it doesn't exist.}
        {--case-insensitive=true : The image / segmentation search will be performed in a case insensitive manner.}
        {--xray=xray.nii.gz : X-Ray Image Name.}
        {--c2=c2.nii.gz : C2 Disc Segmentation filename.}
        {--c3=c3.nii.gz : C3 Disc Segmentation filename.}
        {--c4=c4.nii.gz : C4 Disc Segmentation filename.}
        {--c5=c5.nii.gz : C5 Disc Segmentation filename.}
        {--c6=c6.nii.gz : C6 Disc Segmentation filename.}
        {--c7=c7.nii.gz : C7 Disc Segmentation filename.}
        {--workers=1 : Number of parallel workers. If set to 0 use the number of CPU cores.}
        {--extra-optimizations=false : Generally gives a slightly better result at the cost of long compute time when set to true.}
        {--DEBUG=false : Enable debugging mode.}
    """

    def handle(self):  # type: () -> Optional[int]
        vals, errors = self._validate_options({
            'indir': 'path must-exist',
            'outdir': 'path',
            'case-insensitive': 'bool',
            'extra-optimizations': 'bool',
            'xray': 'str',
            'c2': 'str',
            'c3': 'str',
            'c4': 'str',
            'c5': 'str',
            'c6': 'str',
            'c7': 'str',
            'workers': 'uint',
            'DEBUG': 'bool',
        })

        if len(errors) > 0:
            for k, v in errors.items():
                self.line_error(f'--{k}: {v}')

            return 1

        if vals['workers'] <= 0:
            vals['workers'] = psutil.cpu_count()

        retval = run_pipeline(**vals)

        if retval is not None and retval != 0:
            appname = self.application.config.name
            self.line_error(f'{appname} finished unexpectedly, please fix the errors above and rerun the tool')

    def _validate_options(self, options):
        errors = {}
        retvals = {}
        for option, type in options.items():
            if type == 'path must-exist':
                val, error = self._validate_path(self.option(option), must_exist=True)
            elif type == 'path':
                val, error = self._validate_path(self.option(option), must_exist=False)
            elif type == 'bool':
                val = self.option(option)
                error = None
                if val is not None:
                    val = val.lower().strip() in ['true', 'yes', '1', 'yarr']
            elif type == 'str':
                val = self.option(option)
                error = None
            elif type == 'uint':
                val = self.option(option)
                if val is not None and val.isnumeric():
                    val = int(val)
                else:
                    error = 'Invalid value, must be a non-negative number'

                if val < 0:
                    error = 'Invalid value, must be a non-negative number'

            retvals[option.replace('-', '_')] = val

            if error is not None:
                errors[option] = error

        return retvals, errors

    def _validate_path(self, path, must_exist=True):
        if path is None or path == '' or path.strip() == '':
            return None, 'Invalid path specified'

        path = Path(path)
        if must_exist and not path.exists():
            return path, f'Path {path} does not exist'

        return path, None


def run():
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    package = 'd3tox'
    ver = pkg_resources.require(package)[0].version

    application = Application(name=package, version=ver)

    application.add(PreprocessPipeline())

    application.run()


if __name__ == '__main__':
    run()
