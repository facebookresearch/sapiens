# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import platform
import shutil
import sys
import warnings
from typing import List, Dict, Tuple, Optional, Union
from setuptools import find_packages, setup

try:
    import google.colab  # noqa
    ON_COLAB = True
except ImportError:
    ON_COLAB = False


version_file = 'mmpose/version.py'


def get_version() -> str:
    """Retrieve the package version from the version file.

    Returns:
        str: The version of the package.
    """
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    import sys

    # Return short version for sdist
    if 'sdist' in sys.argv or 'bdist_wheel' in sys.argv:
        return locals()['short_version']
    else:
        return locals()['__version__']


def parse_requirements(fname: str = 'requirements.txt', with_version: bool = True) -> List[str]:
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): Path to the requirements file.
        with_version (bool): If True, include version specifications.

    Returns:
        List[str]: List of requirements items.
    """
    import re
    from os.path import exists

    require_fpath = fname

    def parse_line(line: str) -> Dict[str, Optional[Union[str, Tuple[str, str]]]]:
        """Parse information from a line in a requirements text file."""
        info: Dict[str, Optional[Union[str, Tuple[str, str]]]] = {'line': line}
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            return dict(line=line, package=None, version=None, platform_deps=None, target=target)
        else:
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            elif '@git+' in line:
                info['package'] = line
            else:
                # Remove versioning from the package
                pat = r'(\>=|\==|\>)'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        version, platform_deps = map(str.strip, rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest
                    info['version'] = (op, version)
            if ON_COLAB and info.get('package') == 'xtcocotools':
                # Due to an incompatibility between the Colab platform and the
                # pre-built xtcocotools PyPI package, it is necessary to
                # compile xtcocotools from source on Colab.
                info = dict(
                    line=info['line'],
                    package='xtcocotools@'
                    'git+https://github.com/jin-s13/xtcocoapi'
                )
            return info

    def parse_require_file(fpath: str) -> List[Dict[str, Optional[Union[str, Tuple[str, str]]]]]:
        with open(fpath, 'r') as f:
            return [parse_line(line.strip()) for line in f if line.strip() and not line.startswith('#')]

    def gen_packages_items() -> List[str]:
        if exists(require_fpath):
            packages = []
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    platform_deps = info.get('platform_deps')
                    if platform_deps:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                packages.append(item)
            return packages
        return []

    return gen_packages_items()


def add_mim_extension() -> None:
    """Add extra files that are required to support MIM into the package.

    These files will be added by creating a symlink to the originals if the
    package is installed in `editable` mode (e.g., pip install -e .), or by
    copying from the originals otherwise.
    """
    # Parse installment mode
    if 'develop' in sys.argv:
        mode = 'copy' if platform.system() == 'Windows' else 'symlink'
    elif 'sdist' in sys.argv or 'bdist_wheel' in sys.argv:
        mode = 'copy'
    else:
        return

    filenames = [
        'tools', 'configs', 'demo', 'model-index.yml', 'dataset-index.yml'
    ]
    repo_path = osp.dirname(__file__)
    mim_path = osp.join(repo_path, 'mmpose', '.mim')
    os.makedirs(mim_path, exist_ok=True)

    for filename in filenames:
        src_path = osp.join(repo_path, filename)
        tar_path = osp.join(mim_path, filename)

        if osp.exists(tar_path):
            if osp.isfile(tar_path) or osp.islink(tar_path):
                os.remove(tar_path)
            elif osp.isdir(tar_path):
                shutil.rmtree(tar_path)

        if mode == 'symlink':
            src_relpath = osp.relpath(src_path, osp.dirname(tar_path))
            os.symlink(src_relpath, tar_path)
        elif mode == 'copy':
            if osp.isfile(src_path):
                shutil.copyfile(src_path, tar_path)
            elif osp.isdir(src_path):
                shutil.copytree(src_path, tar_path)
            else:
                warnings.warn(f'Cannot copy file {src_path}.')
        else:
            raise ValueError(f'Invalid mode {mode}')


if __name__ == '__main__':
    add_mim_extension()
    setup(
        name='sapiens_pose',
        version=get_version(),
        description='Sapiens: Foundation for Human Vision Models',
        author='Meta Reality Labs',
        author_email='',
        keywords='computer vision, pose estimation',
        long_description='',
        long_description_content_type='text/markdown',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        include_package_data=True,
        package_data={'mmpose.ops': ['*/*.so']},
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
        url='',
        license='',
        python_requires='>=3.7',
        install_requires=parse_requirements('requirements/runtime.txt'),
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'tests': parse_requirements('requirements/tests.txt'),
            'optional': parse_requirements('requirements/optional.txt'),
            'mim': parse_requirements('requirements/mminstall.txt'),
        },
        zip_safe=False
    )
