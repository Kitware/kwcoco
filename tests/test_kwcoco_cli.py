from os.path import exists
from os.path import join


def test_kwcoco_cli():

    import pytest
    pytest.skip('disable for now')

    import ubelt as ub
    dpath = ub.Path.appdir('kwcoco/test/cli').ensuredir()
    ub.delete(dpath)
    ub.ensuredir(dpath)

    verbose = 3

    cmdkw = dict(verbose=verbose, check=True, cwd=dpath)

    info = ub.cmd('kwcoco --help', **cmdkw)

    info = ub.cmd('kwcoco toydata --dst foo.json', **cmdkw)
    assert exists(join(dpath, 'foo.json'))

    info = ub.cmd('kwcoco stats --src foo.json', **cmdkw)

    info = ub.cmd('kwcoco split --src foo.json --dst1 foo1.json --dst2=foo2.json', **cmdkw)
    assert exists(join(dpath, 'foo1.json'))
    assert exists(join(dpath, 'foo2.json'))

    info = ub.cmd('kwcoco split --src foo1.json --dst1 foo3.json --dst2=foo4.json', **cmdkw)
    assert exists(join(dpath, 'foo3.json'))
    assert exists(join(dpath, 'foo4.json'))

    info = ub.cmd('kwcoco union --src foo3.json foo4.json --dst bar.json', **cmdkw)
    assert exists(join(dpath, 'bar.json'))

    info = ub.cmd('kwcoco show --src foo3.json --dst foo3.png', **cmdkw)  # NOQA
    assert exists(join(dpath, 'foo3.png'))


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/tests/test_kwcoco_cli.py
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
