from _typeshed import Incomplete
from kwcoco import coco_evaluator


class CocoEvalCLIConfig(coco_evaluator.CocoEvalConfig):
    __default__: Incomplete


class CocoEvalCLI:
    name: str
    CLIConfig = CocoEvalCLIConfig

    @classmethod
    def main(cls, cmdline: bool = ..., **kw) -> None:
        ...


def main(cmdline: bool = ..., **kw):
    ...
