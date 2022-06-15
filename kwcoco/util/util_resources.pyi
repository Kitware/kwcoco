from typing import Union


def coerce_num_workers(num_workers: Union[int, str] = 'auto',
                       minimum: int = 0) -> int:
    ...
