# import functools
import ubelt as ub


# @functools.cache
@ub.memoize
def _get_rich_print():
    try:
        import rich
    except ImportError:
        return print
    else:
        return rich.print


def rich_print(*args, **kwargs):
    """
    Does a rich print if available, otherwise fallback to regular print
    """
    print_func = _get_rich_print()
    print_func(*args, **kwargs)
