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

    # Create a list of all submodules with CLI interfaces
    cli_modules = [
        coco_stats,
        coco_union,
        coco_split,
        coco_show,
        coco_toydata,
    ]

    # Create a subparser that uses the first positional argument to run one of
    # the previous CLI interfaces.
    parser = argparse.ArgumentParser(
        description='The Kitware COCO CLI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(help='specify a command to run')

    for cli_module in cli_modules:
        cli_cls = cli_module._CLI
        subconfig = cli_cls.CLIConfig()

        subparser = subparsers.add_parser(
                cli_cls.name, help=subconfig.__class__.__doc__)
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
