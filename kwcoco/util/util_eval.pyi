from typing import Dict
from typing import Any
from typing import List
from typing import Any


class RestrictedSyntaxError(Exception):
    ...


def restricted_eval(expr: str,
                    max_chars: int = ...,
                    local_dict: Dict[str, Any] | None = None,
                    builtins_passlist: List[str] | None = None):
    ...
