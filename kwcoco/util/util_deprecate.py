"""
Deprecation helpers
"""
import ubelt as ub


def deprecated_function_alias(modname, old_name, new_func, deprecate=None, error=None, remove=None):
    """
    Exposes an old deprecated alias of a new prefered function
    """
    new_name = new_func.__name__
    def _deprecated_func_wrapper(*args, **kwargs):
        ub.schedule_deprecation(
            modname=modname, name=old_name, type='function',
            migration=f'Use {new_name} instead',
            deprecate=deprecate, error=error, remove=remove)

        return new_func(*args, **kwargs)
    _deprecated_func_wrapper.__name__ = old_name
    return _deprecated_func_wrapper
