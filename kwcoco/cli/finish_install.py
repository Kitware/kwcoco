#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import scriptconfig as scfg
import ubelt as ub


__dev_notes__ = """
SeeAlso:
    ~/code/graphid/graphid/cli/finish_install.py
    ~/code/kwcoco/kwcoco/cli/finish_install.py
"""


class FinishInstallCLI(scfg.DataConfig):
    """
    Finish the install of kwcoco.

    This is a script that handles install logic that could not be added
    to the setup.py
    """
    __command__ = 'finish_install'

    with_gdal = scfg.Value(True, isflag=True, help='if True, ensure osgeo / gdal is installed')
    with_cv2_headless = scfg.Value(True, isflag=True, help='if True, ensure cv2 is installed')

    strict = scfg.Value(False, isflag=True, help='if True, use strict versions')

    @classmethod
    def main(FinishInstallCLI, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from kwcoco.cli.finish_install import *  # NOQA
            >>> cmdline = 0
            >>> kwargs = dict()
            >>> main(cmdline=cmdline, **kwargs)
        """
        import rich
        import sys
        config = FinishInstallCLI.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + ub.urepr(config, nl=1))

        if config.with_gdal:
            requirements = get_optional_requirements('gdal.txt', config.strict)
            options = [
                '--prefer-binary',
                '--find-links',
                'https://girder.github.io/large_image_wheels',
            ]
            ub.cmd([sys.executable, '-m', 'pip', 'install'] + options + requirements, verbose=3)

        if config.with_cv2_headless:
            # TODO: add logic for fixing a broken cv2 install.
            requirements = get_optional_requirements('headless.txt', strict=config.strict)
            options = [
                '--prefer-binary',
            ]
            ub.cmd([sys.executable, '-m', 'pip', 'install'] + options + requirements, verbose=3)

        want_system_exes = [
            # 'ffmpeg',
            'jq'
            # Todo: breakdown package managers for requested services
            # 'postgresql',
            # 'postgresql-contrib',
        ]
        request_system_executables(want_system_exes)


def request_system_executables(want_system_exes):
    missing = []
    for exe in want_system_exes:
        found = ub.find_path(exe)
        if found is None:
            missing.append(exe)
    if missing:
        print('Warning: missing system packages: {missing}')


def get_optional_requirements(name, strict=False):
    try:
        from kwcoco.rc.registry import requirement_path
    except Exception:
        raise
        requirement_path = None
    req_path = requirement_path(name)
    requirements = parse_requirements(req_path, versions='strict' if strict else 'loose')
    requirements = [line for line in requirements if line.strip()]
    return requirements


def parse_requirements(fname='requirements.txt', versions='loose'):
    """
    Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        versions (bool | str, default=False):
            If true include version specs.
            If strict, then pin to the minimum version.

    Returns:
        List[str]: list of requirements items
    """
    from os.path import join,  dirname, exists
    import re
    import sys

    require_fpath = fname

    def parse_line(line, dpath=''):
        """
        Parse information from a line in a requirements text file

        line = 'git+https://a.com/somedep@sometag#egg=SomeDep'
        line = '-e git+https://a.com/somedep@sometag#egg=SomeDep'
        """
        # Remove inline comments
        comment_pos = line.find(' #')
        if comment_pos > -1:
            line = line[:comment_pos]

        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = join(dpath, line.split(' ')[1])
            for info in parse_require_file(target):
                yield info
        else:
            # See: https://www.python.org/dev/peps/pep-0508/
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                if '--find-links' in line:
                    # setuptools doesnt seem to handle find links
                    line = line.split('--find-links')[0]
                if ';' in line:
                    pkgpart, platpart = line.split(';')
                    # Handle platform specific dependencies
                    # setuptools.readthedocs.io/en/latest/setuptools.html
                    # #declaring-platform-specific-dependencies
                    plat_deps = platpart.strip()
                    info['platform_deps'] = plat_deps
                else:
                    pkgpart = line
                    platpart = None

                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, pkgpart, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        dpath = dirname(fpath)
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line, dpath=dpath):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if versions and 'version' in info:
                    if versions == 'strict':
                        # In strict mode, we pin to the minimum version
                        if info['version']:
                            # Only replace the first >= instance
                            verstr = ''.join(info['version']).replace('>=', '==', 1)
                            parts.append(verstr)
                    else:
                        parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    plat_deps = info.get('platform_deps')
                    if plat_deps is not None:
                        parts.append(';' + plat_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


__cli__ = FinishInstallCLI
main = __cli__.main


if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/kwcoco/cli/finish_install.py
        python -m kwcoco.cli.finish_install
    """
    main()
