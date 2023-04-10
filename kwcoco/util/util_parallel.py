

def coerce_num_workers(num_workers='auto', minimum=0):
    """
    Return some number of CPUs based on a chosen hueristic

    Args:
        num_workers (int | str):
            A special string code, or an exact number of cpus

        minimum (int): minimum workers we are allowed to return

    Returns:
        int : number of available cpus based on request parameters

    CommandLine:
        xdoctest -m kwcoco.util.util_parallel coerce_num_workers

    Example:
        >>> from kwcoco.util.util_parallel import *  # NOQA
        >>> print(coerce_num_workers('all'))
        >>> print(coerce_num_workers('avail'))
        >>> print(coerce_num_workers('auto'))
        >>> print(coerce_num_workers('all-2'))
        >>> print(coerce_num_workers('avail-2'))
        >>> print(coerce_num_workers('all/2'))
        >>> print(coerce_num_workers('min(all,2)'))
        >>> print(coerce_num_workers('[max(all,2)][0]'))
        >>> import pytest
        >>> with pytest.raises(Exception):
        >>>     print(coerce_num_workers('all + 1' + (' + 1' * 100)))
        >>> total_cpus = coerce_num_workers('all')
        >>> assert coerce_num_workers('all-2') == max(total_cpus - 2, 0)
        >>> assert coerce_num_workers('all-100') == max(total_cpus - 100, 0)
        >>> assert coerce_num_workers('avail') <= coerce_num_workers('all')
        >>> assert coerce_num_workers(3) == max(3, 0)
    """
    import numpy as np
    import psutil
    from kwcoco.util.util_eval import restricted_eval

    try:
        num_workers = int(num_workers)
    except Exception:
        pass

    if isinstance(num_workers, str):

        num_workers = num_workers.lower()

        if num_workers == 'auto':
            num_workers = 'avail-2'

        # input normalization
        num_workers = num_workers.replace('available', 'avail')

        local_dict = {}

        # prefix = 'avail'
        if 'avail' in num_workers:
            current_load = np.array(psutil.cpu_percent(percpu=True)) / 100
            local_dict['avail'] = np.sum(current_load < 0.5)
        local_dict['all_'] = psutil.cpu_count()

        if num_workers == 'none':
            num_workers = None
        else:
            expr = num_workers.replace('all', 'all_')
            # limit chars even futher if eval is used
            if 1:
                # Mitigate attack surface by restricting builtin usage
                max_chars = 32
                builtins_passlist = ['min', 'max', 'round', 'sum']
                num_workers = restricted_eval(expr, max_chars, local_dict,
                                              builtins_passlist)
            else:
                # note: eval is not safe, mabye use numexpr instead
                import numexpr
                num_workers = numexpr.evaluate(expr, local_dict=local_dict,
                                               global_dict=local_dict)

    if num_workers is not None:
        num_workers = max(int(num_workers), minimum)

    return num_workers
