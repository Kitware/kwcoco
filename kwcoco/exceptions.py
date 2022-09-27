class AddError(ValueError):
    """
    Generic error when trying to add a category/annotation/image
    """
    pass


class DuplicateAddError(ValueError):
    """
    Error when trying to add a duplicate item
    """
    pass


class InvalidAddError(ValueError):
    """
    Error when trying to invalid data
    """
    pass
