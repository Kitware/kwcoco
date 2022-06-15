from typing import Callable
from _typeshed import Incomplete


class Element(dict):
    __generics__: Incomplete

    def __init__(self,
                 base: dict,
                 options: dict = ...,
                 _magic: Callable = None) -> None:
        ...

    def __call__(self, *args, **kw):
        ...

    def validate(self, instance: dict = ...):
        ...

    def __or__(self, other):
        ...


class ScalarElements:

    @property
    def NULL(self):
        ...

    @property
    def BOOLEAN(self):
        ...

    @property
    def STRING(self):
        ...

    @property
    def NUMBER(self):
        ...

    @property
    def INTEGER(self):
        ...


class QuantifierElements:

    @property
    def ANY(self):
        ...

    def ALLOF(self, *TYPES):
        ...

    def ANYOF(self, *TYPES):
        ...

    def ONEOF(self, *TYPES):
        ...

    def NOT(self, TYPE):
        ...


class ContainerElements:

    def ARRAY(self, TYPE=..., **kw):
        ...

    def OBJECT(self, PROPERTIES=..., **kw):
        ...


class SchemaElements(ScalarElements, QuantifierElements, ContainerElements):
    ...


elem: Incomplete
ALLOF: Incomplete
ANY: Incomplete
ANYOF: Incomplete
ARRAY: Incomplete
BOOLEAN: Incomplete
INTEGER: Incomplete
NOT: Incomplete
NULL: Incomplete
NUMBER: Incomplete
OBJECT: Incomplete
ONEOF: Incomplete
STRING: Incomplete
