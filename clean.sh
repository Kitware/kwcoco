#!/bin/bash
rm -rf htmlcov
rm -rf _cache
rm -rf docs/build
rm -rf .mypy_cache

CLEAN_PYTHON='find . -iname *.pyc -delete ; find . -iname *.pyo -delete ; find . -regex ".*\(__pycache__\|\.py[co]\)" -delete'
bash -c "$CLEAN_PYTHON"
