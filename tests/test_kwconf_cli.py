"""Regression tests for the kwcoco CLI migration to kwconf."""

import importlib

import pytest


from kwcoco.cli.__main__ import CLI_MODULE_NAMES


def test_all_cli_configs_validate_with_kwconf():
    """Catch alias collisions and invalid kwconf field declarations."""
    import kwconf

    for modname in CLI_MODULE_NAMES:
        module = importlib.import_module(f'kwcoco.cli.{modname}')
        cli_cls = module.__cli__
        assert issubclass(cli_cls, kwconf.Config)
        cli_cls.validate()


def test_kwconf_explicit_boolean_negation():
    from kwcoco.cli.coco_conform import CocoConformCLI

    config = CocoConformCLI.cli(
        argv=['input.kwcoco.json', 'output.kwcoco.json', '--ensure_imgsize=false']
    )
    assert config.ensure_imgsize is False


def test_kwconf_special_options_are_enabled(capsys):
    from kwcoco.cli.coco_conform import CocoConformCLI

    # kwconf deliberately keeps Config.argparse() minimal unless
    # special_options=True is passed explicitly. Config.cli() consults the
    # class-level policy, which is the behavior kwcoco relies on.
    assert CocoConformCLI.__special_options__ is True
    parser = CocoConformCLI().argparse(special_options=True)
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert '--config' in option_strings
    assert '--dump' in option_strings
    assert '--dumps' in option_strings

    with pytest.raises(SystemExit) as exc_info:
        CocoConformCLI.cli(argv=['--dumps'])
    assert exc_info.value.code == 0
    assert 'ensure_imgsize:' in capsys.readouterr().out


def test_kwconf_short_alias_without_embedded_dash():
    from kwcoco.cli.coco_move_assets import CocoMoveAssetsCLI

    config = CocoMoveAssetsCLI.cli(argv=['-n'])
    assert config.dry is True


def test_kwcoco_root_modal_builds_and_shows_help():
    from kwcoco.cli import __main__

    with pytest.raises(SystemExit) as exc_info:
        __main__.main(argv=['--help'])
    assert exc_info.value.code == 0


def test_kwcoco_root_registration_matches_cli_modules():
    """Guard against shipped CLI modules silently disappearing from ModalCLI."""
    from pathlib import Path
    from kwcoco.cli import __main__
    import kwcoco.cli

    cli_dpath = Path(kwcoco.cli.__file__).parent
    discovered = set()
    for fpath in cli_dpath.glob('*.py'):
        if fpath.name.startswith('_'):
            continue
        module = importlib.import_module(f'kwcoco.cli.{fpath.stem}')
        if hasattr(module, '__cli__'):
            discovered.add(fpath.stem)
    assert discovered == set(CLI_MODULE_NAMES)

    with pytest.raises(SystemExit) as exc_info:
        __main__.main(argv=['move_assets', '--help'])
    assert exc_info.value.code == 0


def test_kwconf_validate_workers_is_not_boolean():
    from kwcoco.cli.coco_validate import CocoValidateCLI

    config = CocoValidateCLI.cli(argv=['demo.kwcoco.json', '--workers=3'])
    assert config.workers == 3


def test_move_assets_requires_real_src_dst_values():
    from kwcoco.cli.coco_move_assets import CocoMoveAssetsCLI

    config = CocoMoveAssetsCLI.cli(argv=[])
    assert config.src is None
    assert config.dst is None


def test_kwconf_dual_bare_and_explicit_value_options():
    """Preserve scriptconfig-era options that were both flags and values."""
    from kwcoco.cli.coco_info import CocoInfoCLI
    from kwcoco.cli.coco_split import CocoSplitCLI
    from kwcoco.cli.coco_visualize_videos import CocoVisualizeVideosCLI

    assert CocoInfoCLI.cli(argv=['demo.json', '--show_images']).show_images is True
    assert CocoInfoCLI.cli(argv=['demo.json', '--show_images=3']).show_images == 3
    assert CocoVisualizeVideosCLI.cli(argv=['--verbose']).verbose == 1
    assert CocoVisualizeVideosCLI.cli(argv=['--verbose=3']).verbose == 3
    assert CocoSplitCLI.cli(argv=['--num_write=3']).num_write == 3
