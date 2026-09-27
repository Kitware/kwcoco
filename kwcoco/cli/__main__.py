from __future__ import annotations

#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
import sys
import ubelt as ub


CLI_MODULE_NAMES = [
    'coco_stats',
    'coco_union',
    'coco_split',
    'coco_show',
    'coco_toydata',
    'coco_eval',
    'coco_conform',
    'coco_modify_categories',
    'coco_reroot',
    'coco_fixup',
    'coco_move',
    'coco_move_assets',
    'coco_validate',
    'coco_subset',
    'coco_grab',
    'coco_info',
    'coco_plot_stats',
    'coco_visualize_videos',
    'find_unregistered_images',
    'finish_install',
]


def main(argv=True, **kw):
    """
    kw = dict(command='stats')
    argv = False
    """
    module_lut = {}
    for name in CLI_MODULE_NAMES:
        mod = ub.import_module_from_name('kwcoco.cli.{}'.format(name))
        module_lut[name] = mod

    # Create a list of all submodules with CLI interfaces
    cli_modules = list(module_lut.values())

    # Create a subparser that uses the first positional argument to run one of
    # the previous CLI interfaces.
    import os

    KWCOCO_LOOSE_CLI = os.environ.get('KWCOCO_LOOSE_CLI', '')

    import kwconf

    import kwcoco

    modal = kwconf.ModalCLI(
        description=ub.codeblock(
            """
        The Kitware COCO CLI
        """
        ),
        version=kwcoco.__version__,
    )

    for cli_module in cli_modules:
        cli_config = None
        if hasattr(cli_module, '__cli__'):
            # New way
            cli_config = cli_module.__cli__
        else:
            raise NotImplementedError(
                f'modules must define the __cli__ attribute to be registered. Failed on {cli_module}'
            )

        # Update configs to have aliases / commands attributes
        # cli_modname = cli_module.__name__
        # cli_rel_modname = cli_modname.split('.')[-1]
        cmdname_aliases = ub.oset()
        alias = getattr(cli_module, '__alias__', getattr(cli_config, '__alias__', []))
        if isinstance(alias, str):
            alias = [alias]
        command = getattr(
            cli_module, '__command__', getattr(cli_config, '__command__', None)
        )
        if command is not None:
            cmdname_aliases.add(command)
        cmdname_aliases.update(alias)
        # cmdname_aliases.update(cmd_alias.get(cli_modname, []) )
        cmdname_aliases.add(cli_config.__command__)
        primary_cmdname = cmdname_aliases[0]
        secondary_cmdnames = cmdname_aliases[1:]
        cli_config.__command__ = primary_cmdname
        cli_config.__alias__ = secondary_cmdnames
        modal.register(cli_config)

    EASTER = 1
    if EASTER:
        if len(sys.argv) == 2 and sys.argv[1] in {'boid', 'boids'}:
            from kwcoco.demo.boids import _yeah_boid

            _yeah_boid()
            sys.exit(0)

    ret = modal.run(argv=argv, strict=not KWCOCO_LOOSE_CLI)
    return ret


if __name__ == '__main__':
    """
    CommandLine:
        python -m kwcoco --help
        python -m kwcoco.coco_stats
    """
    sys.exit(main())
