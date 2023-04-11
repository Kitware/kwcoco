"""
Defines a safer eval function
"""


class RestrictedSyntaxError(Exception):
    """
    An exception raised by restricted_eval if a disallowed expression is given
    """
    pass


def restricted_eval(expr, max_chars=32, local_dict=None, builtins_passlist=None):
    """
    A restricted form of Python's eval that is meant to be slightly safer

    Args:
        expr (str): the expression to evaluate
        max_char (int): expression cannot be more than this many characters
        local_dict (Dict[str, Any]): a list of variables allowed to be used
        builtins_passlist (List[str] | None) : if specified, only allow use of certain builtins

    References:
        https://realpython.com/python-eval-function/#minimizing-the-security-issues-of-eval

    Notes:
        This function may not be safe, but it has as many mitigation measures
        that I know about. This function should be audited and possibly made
        even more restricted. The idea is that this should just be used to
        evaluate numeric expressions.

    Example:
        >>> from kwcoco.util.util_eval import *  # NOQA
        >>> builtins_passlist = ['min', 'max', 'round', 'sum']
        >>> local_dict = {}
        >>> max_chars = 32
        >>> expr = 'max(3 + 2, 9)'
        >>> result = restricted_eval(expr, max_chars, local_dict, builtins_passlist)
        >>> expr = '3 + 2'
        >>> result = restricted_eval(expr, max_chars, local_dict, builtins_passlist)
        >>> expr = '3 + 2'
        >>> result = restricted_eval(expr, max_chars)
        >>> import pytest
        >>> with pytest.raises(RestrictedSyntaxError):
        >>>     expr = 'max(a + 2, 3)'
        >>>     result = restricted_eval(expr, max_chars, dict(a=3))
    """
    import builtins
    if len(expr) > max_chars:
        raise RestrictedSyntaxError(
            'num-workers-hueristic should be small text. '
            'We want to disallow attempts at crashing python '
            'by feeding nasty input into eval. But this may still '
            'be dangerous.'
        )
    if local_dict is None:
        local_dict = {}

    if builtins_passlist is None:
        builtins_passlist = []

    _builtins_passlist = set(builtins_passlist)
    allowed_builtins = {k: v for k, v in builtins.__dict__.items()
                        if k in _builtins_passlist}

    local_dict['__builtins__'] = allowed_builtins
    allowed_names = list(allowed_builtins.keys()) + list(local_dict.keys())
    code = compile(expr, "<string>", "eval")
    # Step 3
    for name in code.co_names:
        if name not in allowed_names:
            raise RestrictedSyntaxError(f"Use of {name} not allowed")
    result = eval(code, local_dict, local_dict)
    return result
