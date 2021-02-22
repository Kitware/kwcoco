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
    from kwcoco.cli import coco_eval
    from kwcoco.cli import coco_conform
    from kwcoco.cli import coco_modify_categories
    from kwcoco.cli import coco_reroot
    from kwcoco.cli import coco_validate

    # Create a list of all submodules with CLI interfaces
    cli_modules = [
        coco_stats,
        coco_union,
        coco_split,
        coco_show,
        coco_reroot,
        coco_toydata,
        coco_conform,
        coco_eval,
        coco_modify_categories,
        coco_validate,
    ]

    # Create a subparser that uses the first positional argument to run one of
    # the previous CLI interfaces.

    class RawDescriptionDefaultsHelpFormatter(
            argparse.RawDescriptionHelpFormatter,
            argparse.ArgumentDefaultsHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description='The Kitware COCO CLI',
        formatter_class=RawDescriptionDefaultsHelpFormatter,
    )
    parser.add_argument('--version', action='store_true',
                        help='show version number and exit')
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

    if 0:
        """
        Debugging positional or keyword args

            python -m kwcoco.cli.coco_stats special:shapes8

            python -m kwcoco.cli.coco_stats --src=special:shapes8

            >>> kw = {'src': 'special:shapes8'}
            >>> cmdline = False
            >>> cls = CocoStatsCLI

            python -c "from kwcoco.cli.coco_stats import *; print(CocoStatsCLI.CLIConfig()._read_argv())" --src foo bar
            python -c "from kwcoco.cli.coco_stats import *; print(CocoStatsCLI.CLIConfig()._read_argv())" a --basic=True baz biz --src f a a

        """
        for action in parser._actions:
            print('action = {!r}'.format(action))
            pass
        for sub in parser._subparsers:
            parser._subparsers._actions
            pass

    ns = parser.parse_known_args()[0]
    # print('ns = {!r}'.format(ns))

    # Execute the subcommand without additional CLI parsing
    kw = ns.__dict__

    if kw.pop('version'):
        import kwcoco
        print(kwcoco.__version__)
        return 0

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
    """
    sys.exit(main())
