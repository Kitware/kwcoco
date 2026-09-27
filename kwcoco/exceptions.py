from __future__ import annotations


class AddError(ValueError):
    """
    Generic error when trying to add a category/annotation/image
    """

    pass


class DuplicateAddError(AddError):
    """
    Error when trying to add a duplicate item
    """

    pass


class InvalidAddError(AddError):
    """
    Error when trying to invalid data
    """

    pass
