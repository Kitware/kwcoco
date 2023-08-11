from typing import Any
from typing import Tuple
from typing import Dict
import ubelt as ub
from collections.abc import Generator
from scriptconfig.dict_like import DictLike as DictLike_


class DictLike(ub.NiceRepr):

    def getitem(self, key: Any) -> Any:
        ...

    def setitem(self, key: Any, value: Any) -> None:
        ...

    def delitem(self, key: Any) -> None:
        ...

    def keys(self) -> Generator[Any, None, None]:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self):
        ...

    def __contains__(self, key: Any) -> bool:
        ...

    def __delitem__(self, key: Any):
        ...

    def __getitem__(self, key: Any) -> Any:
        ...

    def __setitem__(self, key: Any, value: Any):
        ...

    def items(self) -> Generator[Tuple[Any, Any], None, None]:
        ...

    def values(self) -> Generator[Any, None, None]:
        ...

    def copy(self) -> Dict:
        ...

    def to_dict(self) -> Dict:
        ...

    asdict = to_dict

    def update(self, other) -> None:
        ...

    def get(self, key: Any, default: Any | None = None) -> Any:
        ...


class DictProxy(DictLike_):

    def __getitem__(self, key):
        ...

    def __setitem__(self, key, value) -> None:
        ...

    def keys(self):
        ...

    def __json__(self):
        ...
