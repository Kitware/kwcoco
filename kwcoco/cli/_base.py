"""Shared configuration policy for kwcoco command-line interfaces."""
from __future__ import annotations

import kwconf


class KWCocoConfig(kwconf.Config):
    """Base class that preserves kwcoco's config-file CLI conveniences."""

    __special_options__ = True
