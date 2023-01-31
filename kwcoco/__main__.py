#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
if __name__ == '__main__':
    """
    CommandLine:
        python -m kwcoco --help
        python -m kwcoco.coco_stats
        python ~/code/kwcoco/coco_cli/__main__.py
    """
    from kwcoco.cli.__main__ import main
    main()
