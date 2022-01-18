
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
