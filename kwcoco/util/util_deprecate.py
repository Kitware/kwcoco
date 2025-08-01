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
        Dict[str, Any]: a map from the new cannonical names to the values

    Example:
        >>> from kwcoco.util.util_deprecate import *  # NOQA
        >>> # Simple case
        >>> # We have a function that currently takes an argument `a`
        >>> # but we want to change the signature name to more readable `array`
        >>> # To do this we need to modify the function signature so
        >>> # the old explicit argument defaults to None and then add **kwargs
        >>> # Then in the body we call `migrate_argnames`, which will
        >>> # map the inputs to the new cannonical name, and ensure that
        >>> # no extra kwargs are given.
        >>> def myfunc(a=None, **kwargs):
        >>>     explicit_args = dict(a=a)
        >>>     aliases = {
        >>>         'array': 'a',
        >>>     }
        >>>     result = migrate_argnames(
        >>>         aliases,
        >>>         explicit_args,
        >>>         kwargs,
        >>>     )
        >>>     return result
        >>> # Now we can call the function using positional args
        >>> print(myfunc([1, 2]))
        >>> # Use the old name as a keyword arg
        >>> print(myfunc(a=[1, 2]))
        >>> # Or use the new name as a keyword arg
        >>> print(myfunc(array=[1, 2]))
        >>> import pytest
        >>> # We cannot specify both old and new values
        >>> with pytest.raises(ValueError):
        >>>     myfunc(a=[1, 2], array=[1, 2])
        >>> # We cannot give undefined arguments
        >>> with pytest.raises(TypeError):
        >>>     print(myfunc(does_not_exist=[2]))

    Example:
        >>> from kwcoco.util.util_deprecate import *  # NOQA
        >>> # Demonstrate a function where we have an existing signature with
        >>> # gids and vidids, but we plan to switch to image_ids and video_ids.
        >>> # We also want to still support an old argument _flag1, even though
        >>> # we have already "switched" to a new name: flag1.
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

    Notes:
        Possible improvements:
            - [ ] add a fast path when the cannonical key is given
            - [ ] add the ability to mark arguments are required
            - [ ] get caller information for better warning information
            - [ ] use ub.NoParam to allow None as a valid value.
            - [ ] allow cannon args to specify a default if it is not specified
    """
    import warnings
    # Ensure all values are List[str]
    aliases = {k: [vs] if isinstance(vs, str) else vs
               for k, vs in aliases.items()}

    cannonical = {}

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
        cannonical[cannon_key] = given_value

    # Catch any unexpected keyword arguments in kwargs
    if len(kwargs):
        # TODO: option to grab or specify the name of the caller?
        # should we use the schedule deprecation here as well?
        bad_key = list(kwargs)[0]
        raise TypeError(f'<calling function> got an unexpected keyword argument {bad_key!r}')
    return cannonical
