import sys


def main(cmdline=True, **kw):
    """
    kw = dict(command='stats')
    cmdline = False
    """
    import argparse
    from kwcoco.cli import coco_stats
    from kwcoco.cli import coco_union
    from kwcoco.cli import coco_split
    from kwcoco.cli import coco_show
    from kwcoco.cli import coco_toydata
    # from kwcoco.cli import coco_rebase

    # Create a list of all submodules with CLI interfaces
    cli_modules = [
        coco_stats,
        coco_union,
        coco_split,
        coco_show,
        # coco_rebase,
        coco_toydata,
    ]

    # Create a subparser that uses the first positional argument to run one of
    # the previous CLI interfaces.
    parser = argparse.ArgumentParser(
        description='The Kitware COCO CLI',
        # formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(help='specify a command to run')

    for cli_module in cli_modules:
        cli_cls = cli_module._CLI
        subconfig = cli_cls.CLIConfig()

        # TODO: make subparser.add_parser args consistent with what
        # scriptconfig generates when parser=None
        if hasattr(subconfig, '_parserkw'):
            parserkw = subconfig._parserkw()
        else:
            # for older versions of scriptconfig
            parserkw = dict(
                description=subconfig.__class__.__doc__
            )
        parserkw['help'] = parserkw['description']
        subparser = subparsers.add_parser(cli_cls.name, **parserkw)
        subparser = subconfig.argparse(subparser)
        subparser.set_defaults(main=cli_cls.main)

    ns = parser.parse_args()

    # Execute the subcommand without additional CLI parsing
    kw = ns.__dict__
    main = kw.pop('main', None)
    if main is None:
        parser.print_help()
        raise ValueError('no command given')
        return 1

    try:
        ret = main(cmdline=False, **kw)
    except Exception as ex:
        print('ERROR ex = {!r}'.format(ex))
        raise
        return 1
    else:
        if ret is None:
            ret = 0
        return ret


if __name__ == '__main__':
    """
    CommandLine:
        python -m kwcoco --help
        python -m kwcoco.coco_stats

        python ~/code/ndsampler/coco_cli/__main__.py
    """
    sys.exit(main())
