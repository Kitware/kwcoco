# Should likely just ensure kwutil.Json supports different backends and use
# that instead.
import os
import json as pjson
from packaging.version import parse as Version
from types import ModuleType

try:
    import ujson
except ImportError:
    ujson = None  # type: ignore

KWCOCO_USE_UJSON = bool(os.environ.get('KWCOCO_USE_UJSON'))

json_r: ModuleType
json_w: ModuleType

if (
    ujson is not None
    and Version(ujson.__version__) >= Version('5.2.0')
    and KWCOCO_USE_UJSON
):
    json_r = ujson
    json_w = pjson
else:
    json_r = pjson
    json_w = pjson
