
class SupressPrint(object):
    """
    Temporarily replace the print function in a module with a noop

    Args:
        *mods: the modules to disable print in
        **kw: only accepts "enabled"
            enabled (bool, default=True): enables or disables this context
    """
    def __init__(self, *mods, **kw):
        enabled = kw.get('enabled', True)
        self.mods = mods
        self.enabled = enabled
        self.oldprints = {}

    def __enter__(self):
        if self.enabled:
            for mod in self.mods:
                oldprint = getattr(self.mods, 'print', print)
                self.oldprints[mod] = oldprint
                mod.print = lambda *args, **kw: None

    def __exit__(self, a, b, c):
        if self.enabled:
            for mod in self.mods:
                mod.print = self.oldprints[mod]


class Reloadable(type):
    """
    This is a metaclass that overrides the behavior of isinstance and
    issubclass when invoked on classes derived from this such that they only
    check that the module and class names agree, which are preserved through
    module reloads, whereas class instances are not.

    This is useful for interactive develoment, but should be removed in
    production.

    Example:
        >>> from kwcoco.util.util_monkey import *  # NOQA
        >>> # Illustrate what happens with a reload when using this utility
        >>> # versus without it.
        >>> class Base1:
        >>>     ...
        >>> class Derived1(Base1):
        >>>     ...
        >>> @Reloadable.add_metaclass
        >>> class Base2:
        >>>     ...
        >>> class Derived2(Base2):
        >>>     ...
        >>> inst1 = Derived1()
        >>> inst2 = Derived2()
        >>> assert isinstance(inst1, Derived1)
        >>> assert isinstance(inst2, Derived2)
        >>> # Simulate reload
        >>> class Base1:
        >>>     ...
        >>> class Derived1(Base1):
        >>>     ...
        >>> @Reloadable.add_metaclass
        >>> class Base2:
        >>>     ...
        >>> class Derived2(Base2):
        >>>     ...
        >>> assert not isinstance(inst1, Derived1)
        >>> assert isinstance(inst2, Derived2)
    """

    def __subclasscheck__(cls, sub):
        """
        Is ``sub`` a subclass of ``cls``
        """
        cls_mod_name = (cls.__module__, cls.__name__)
        for b in sub.__mro__:
            b_mod_name = (b.__module__, b.__name__)
            if cls_mod_name == b_mod_name:
                return True

    def __instancecheck__(cls, inst):
        """
        Is ``inst`` an instance of ``cls``
        """
        return cls.__subclasscheck__(inst.__class__)

    @classmethod
    def add_metaclass(metaclass, cls):
        """
        Class decorator for creating a class with this as a metaclass
        """
        orig_vars = cls.__dict__.copy()
        slots = orig_vars.get('__slots__')
        if slots is not None:
            if isinstance(slots, str):
                slots = [slots]
            for slots_var in slots:
                orig_vars.pop(slots_var)
        orig_vars.pop('__dict__', None)
        orig_vars.pop('__weakref__', None)
        if hasattr(cls, '__qualname__'):
            orig_vars['__qualname__'] = cls.__qualname__
        return metaclass(cls.__name__, cls.__bases__, orig_vars)

    @classmethod
    def developing(metaclass, cls):
        """
        Like add_metaclass, but warns the user that they are developing.
        This helps remind them to remove this in production
        """
        import warnings
        warnings.warn(f'Adding the Reloadable metaclass to {cls}. Dont forget to remove')
        return metaclass.add_metaclass(cls)
