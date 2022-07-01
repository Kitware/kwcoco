from numpy import ndarray
from typing import Dict
from typing import Any
from _typeshed import Incomplete
from typing import Any

profile: Incomplete


def dequantize(quant_data: ndarray, quantization: Dict[str, Any]) -> ndarray:
    ...


def quantize_float01(imdata,
                     old_min: int = ...,
                     old_max: int = ...,
                     quantize_dtype=...):
    ...
