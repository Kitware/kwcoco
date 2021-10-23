"""
Truncate utility based on python-slugify.

https://pypi.org/project/python-slugify/1.2.2/
"""


def _trunc_op(string, max_length, trunc_loc):
    """
    Example:
        >>> from kwcoco.util.util_truncate import _trunc_op
        >>> string = 'DarnOvercastSculptureTipperBlazerConcaveUnsuitedDerangedHexagonRockband'
        >>> max_length = 16
        >>> trunc_loc = 0.5
        >>> _trunc_op(string, max_length, trunc_loc)

        >>> from kwcoco.util.util_truncate import _trunc_op
        >>> max_length = 16
        >>> string = 'a' * 16
        >>> _trunc_op(string, max_length, trunc_loc)

        >>> string = 'a' * 17
        >>> _trunc_op(string, max_length, trunc_loc)
    """
    import ubelt as ub
    import numpy as np
    total_len = len(string)
    mid_pos = int(total_len * trunc_loc)

    num_remove = max(total_len - max_length, 0)
    if num_remove > 0:
        recommend = min(max(4, int(np.ceil(np.log(max(1, num_remove))))), 32)
        hash_len = min(max_length, min(num_remove, recommend))
        num_insert = hash_len + 2

        actual_remove = num_remove + num_insert

        low_pos = max(0, (mid_pos - (actual_remove) // 2))
        high_pos = min(total_len, (mid_pos + (actual_remove) // 2))
        if low_pos <= 0:
            n_extra = actual_remove - (high_pos - low_pos)
            high_pos += n_extra
        if high_pos >= total_len:
            n_extra = actual_remove - (high_pos - low_pos)
            low_pos -= n_extra

        really_removed = (high_pos - low_pos)
        high_pos += (really_removed - actual_remove)

        begin = string[:low_pos]
        mid = string[low_pos:high_pos]
        end = string[high_pos:]

        mid = ub.hash_data(string)[0:hash_len]
        trunc_text = ''.join([begin, '~', mid, '~', end])
    else:
        trunc_text = string
    return trunc_text


def smart_truncate(string, max_length=0, separator=' ', trunc_loc=0.5):
    """
    Truncate a string.
    :param string (str): string for modification
    :param max_length (int): output string length
    :param word_boundary (bool):
    :param save_order (bool): if True then word order of output string is like input string
    :param separator (str): separator between words
    :param trunc_loc (float): fraction of location where to remove the text
    :return:
    """
    string = string.strip(separator)

    if not max_length:
        return string

    if len(string) < max_length:
        return string

    truncated = _trunc_op(string, max_length, trunc_loc)
    return truncated.strip(separator)
