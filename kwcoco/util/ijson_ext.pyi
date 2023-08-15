from _typeshed import Incomplete
from collections.abc import Generator
from ijson import common
from typing import Any

LEXEME_RE: Incomplete
UNARY_LEXEMES: Incomplete
EOF: Incomplete


class UnexpectedSymbol(common.JSONError):

    def __init__(self, symbol, pos) -> None:
        ...


def utf8_encoder(target) -> Generator[None, Any, None]:
    ...


def Lexer(target) -> Generator[None, Any, None]:
    ...


inf: Incomplete


def parse_value(target, multivalue, use_float) -> Generator[None, Any, None]:
    ...


def parse_string(symbol):
    ...


def basic_parse_basecoro(target,
                         multiple_values: bool = ...,
                         allow_comments: bool = ...,
                         use_float: bool = ...):
    ...
