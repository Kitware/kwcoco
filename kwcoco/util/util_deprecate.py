"""
Deprecation helpers
"""
import ubelt as ub


def deprecated_function_alias(modname, old_name, new_func, deprecate=None, error=None, remove=None):
    """
    Exposes an old deprecated alias of a new preferred function
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


def migrate_argnames(aliases, explicit_args, kwargs, warn_non_cannon=False):
    """
    Initial pass at a function to help migrate a function to a keyword argument
    signature while maintaining backwards compatibility.

    Do this by aliasing the arguments.

    A warning is given if both new and old args are specified.

    Args:
        aliases (Dict[str, str | list[str]]):
            A dictionary where keys are canonical argument names and values are
            lists of aliases.  Canonical names can appear in either
            `explicit_args` or `kwargs`.

        explicit_args (Dict[str, Any]):
            The names and values of the args that are explicitly defined in the
            function signature.  If the values of any of these are none, then
            the aliased version is used if it exists in kwargs.

        kwargs (Dict[str, Any]):
            The keyword arguments passed to the function, which may contain either canonical names or aliases.

        warn_non_cannon (bool, default=False):
            If True, a warning will be issued whenever an non-cannonical
            argument (from the aliases) is used in `kwargs`.

    Returns:
        Dict[str, Any]: a map from the new names to the official values

    Example:
        >>> from kwcoco.util.util_deprecate import *  # NOQA
        >>> def myfunc(gid=None, flag1=False, vidids=None, flag2=True, **kwargs):
        >>>     explicit_args = dict(gid=gid, vidids=vidids)
        >>>     aliases = {
        >>>         'image_id': 'gid',
        >>>         'video_ids': 'vidids',
        >>>         'flag1': '_flag1',
        >>>     }
        >>>     result = migrate_argnames(
        >>>         aliases,
        >>>         explicit_args,
        >>>         kwargs,
        >>>     )
        >>>     return result
        >>> print(myfunc(gid=1))
        >>> print(myfunc(image_id=1))
        >>> print(myfunc(video_ids=[2]))
        >>> print(myfunc(vidids=[2]))
        >>> print(myfunc(vidids=[2], image_id=2))
        >>> print(myfunc(video_ids=[2], gid=2))
        >>> import pytest
        >>> # No unused arguments
        >>> with pytest.raises(TypeError):
        >>>     print(myfunc(vidid=[2]))
        >>> # Should not specify more than one name for a cannon key
        >>> with pytest.raises(ValueError):
        >>>     print(myfunc(gid=1, image_id=2))
    """
    import warnings
    # Ensure all values are List[str]
    aliases = {k: [vs] if isinstance(vs, str) else vs
               for k, vs in aliases.items()}

    result = {}

    # Loop over each canonical argument (cannon_key) and its alternative names (alts)
    for cannon_key, alts in aliases.items():
        candidates = {}
        cannon_value = explicit_args.get(cannon_key, None)
        if cannon_value is None:
            cannon_value = kwargs.pop(cannon_key, None)

        # TODO: can we add a fast path?
        candidates[cannon_key] = cannon_value
        for alt_key in alts:
            alt_value = explicit_args.get(alt_key, None)
            if alt_value is None:
                alt_value = kwargs.pop(alt_key, None)
            candidates[alt_key] = alt_value

        # Ensure only one non-None candidate is selected, or use default (None)
        nonnull_candidates = {k: v for k, v in candidates.items() if v is not None}
        if len(nonnull_candidates) > 1:
            raise ValueError(f'More than value was specified for {cannon_key}: {nonnull_candidates}')
        elif len(nonnull_candidates) == 1:
            given_key, given_value = next(iter(nonnull_candidates.items()))
            if warn_non_cannon:
                if given_key != cannon_key:
                    warnings.warn(f"Old argument '{given_key}' used, use new canonical name '{cannon_key}' instead.")
        else:
            given_value = candidates[cannon_key]
        result[cannon_key] = given_value

    # Catch any unexpected keyword arguments in kwargs
    if len(kwargs):
        bad_key = list(kwargs)[0]
        raise TypeError(f'<calling function> got an unexpected keyword argument {bad_key!r}')
    return result
