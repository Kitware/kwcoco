#!/usr/bin/env python
import ubelt as ub
import scriptconfig as scfg


# Code to help generate the CLI from keyword arguments to CocoDataset.validate
__autogen_cli_args__ = r"""
import kwcoco
from xdoctest.docstr import docscrape_google
blocks = docscrape_google.split_google_docblocks(kwcoco.CocoDataset.validate.__doc__)
argblock = dict(blocks)['Args'][0]
found = None
for arg in list(docscrape_google.parse_google_argblock(argblock, clean_desc=False)):
    if arg['name'].startswith('**'):
        found = arg
        break
subdesc = ub.codeblock(found['desc'])
# Sub parsing of kwargs
sub_parts = list(docscrape_google.parse_google_argblock(subdesc))
from vimtk._dirty import format_multiple_paragraph_sentences
for part in sub_parts:
    default = part['type'].split('=')[1]
    line1 = f"'{part['name']}': scfg.Value({default}, help=ub.paragraph("
    kwargs = {}
    wrapped_desc = format_multiple_paragraph_sentences(part['desc'], **kwargs)
    sq = chr(39)
    tsq = sq * 3
    line2 = f'    {tsq}'
    line3 = ub.indent(wrapped_desc)
    line4 = f'    {tsq})),'
    lines = [line1, line2, line3, line4]
    print('\n'.join(lines))
    print('')
"""


class CocoValidateCLI(scfg.DataConfig):
    """
    Validates that a coco file satisfies expected properties.

    Checks that a coco file conforms to the json schema, that assets
    exist, and that other expected properties are satisfied.

    This also has the ability to fix corrupted assets by removing them, but
    that functionality may be moved to a new command in the future.
    """
    __command__ = 'validate'

    src = scfg.Value(None, position=1, help='path to datasets', nargs='+')

    schema = scfg.Value(True, isflag=True, help='if True, validate the json-schema')

    unique = scfg.Value(True, isflag=True, help='if True, validate unique secondary keys')

    missing = scfg.Value(True, isflag=True, help='if True, validate registered files exist')

    corrupted = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            if True, validate data in registered files
            '''))

    channels = scfg.Value(True, isflag=True, help=ub.paragraph(
            '''
            if True, validate that channels in auxiliary/asset items are
            all unique.
            '''))

    require_relative = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            if True, causes validation to fail if paths are non-
            portable, i.e. all paths must be relative to the bundle
            directory. if>0, paths must be relative to bundle root.
            if>1, paths must be inside bundle root.
            '''))

    img_attrs = scfg.Value('warn', help=ub.paragraph(
            '''
            if truthy, check that image attributes contain width and
            height entries. If 'warn', then warn if they do not exist.
            If 'error', then fail.
            '''))

    verbose = scfg.Value(1, help='verbosity flag')

    fastfail = scfg.Value(False, isflag=True, help='if True raise errors immediately')

    workers = scfg.Value(0, isflag=True, help=ub.paragraph(
            '''
            number of workers for checks that support parallelization
            '''))

    # TODO: Move these to a different tool. This should only validate,
    # not fix anything.
    # TODO: See new coco_fixup.py script and use that instead.
    fix = scfg.Value(None, help=ub.paragraph(
            '''
            Code indicating strategy to attempt to fix the dataset. If
            None, do nothing. If remove, removes missing / corrupted
            images. Other strategies may be added in the future. This is
            a heuristic and does not always work. dst must be specified.
            And only one src dataset can be given.

            DEPRECATED. Use kwcoco fixup instead.
            '''))

    dst = scfg.Value(None, help=ub.paragraph(
            '''
            Location to write a "fixed" coco file if a fix strategy is
            given.
            '''))

    __epilog__ = """
    Example Usage:
        kwcoco toydata --dst foo.json --key=special:shapes8
        kwcoco validate --src=foo.json --corrupted=True
    """

    @classmethod
    def main(cls, cmdline=True, **kw):
        """
        Example:
            >>> from kwcoco.cli.coco_validate import *  # NOQA
            >>> kw = {'src': 'special:shapes8'}
            >>> cmdline = False
            >>> cls = CocoValidateCLI
            >>> cls.main(cmdline, **kw)
        """
        import kwcoco
        config = cls.cli(data=kw, cmdline=cmdline, strict=True)
        print('config = {}'.format(ub.urepr(config, nl=1)))

        if config['src'] is None:
            raise Exception('must specify source: {}'.format(config['src']))

        if isinstance(config['src'], str):
            fpaths = [config['src']]
        else:
            fpaths = config['src']

        if config['dst']:
            if len(fpaths) != 1:
                raise Exception('can only specify 1 dataset in fix mode')

        fix_strat = set()
        if config['fix'] is not None:
            fix_strat = {c.lower() for c in config['fix'].split('+')}

        fpath_to_errors = {}
        for fpath in ub.ProgIter(fpaths, desc='reading datasets', verbose=1):
            print('reading fpath = {!r}'.format(fpath))
            dset = kwcoco.CocoDataset.coerce(fpath)

            config_ = ub.dict_diff(config, {'src', 'dst', 'fix'})
            result = dset.validate(**config_)

            if 'missing' in result:
                if 'remove' in fix_strat:
                    missing = result['missing']
                    bad_gids = [t[2] for t in missing]
                    status = dset.remove_images(bad_gids, verbose=1)
                    print('status = {}'.format(ub.urepr(status, nl=1)))

            if 'corrupted' in result:
                if 'remove' in fix_strat:
                    corrupted = result['corrupted']
                    bad_gids = [t[2] for t in corrupted]
                    status = dset.remove_images(bad_gids, verbose=1)
                    print('status = {}'.format(ub.urepr(status, nl=1)))

            if config['dst']:
                if len(fpaths) != 1:
                    raise Exception('can only specify 1 dataset in fix mode')
                dset.dump(config['dst'], newlines=True)

            errors = result['errors']
            fpath_to_errors[fpath] = errors

        has_errors = any(ub.flatten(fpath_to_errors.values()))
        if has_errors:
            errmsg = ub.urepr(fpath_to_errors, nl=1)
            print('fpath_to_errors = {}'.format(errmsg))
            raise Exception(errmsg)


__cli__ = CocoValidateCLI

if __name__ == '__main__':
    """
    CommandLine:
        python -m kwcoco.cli.coco_stats --src=special:shapes8
    """
    __cli__.main()
