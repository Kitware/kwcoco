"""
The ChannelSpec and FusedChannelSpec represent a set of channels or bands in an
image. This could be as simple as ``red|green|blue``, or more complex like:
``red|green|blue|nir|swir16|swir22``.

This functionality has been moved to "delayed_image".
"""
from delayed_image.channel_spec import FusedChannelSpec  # NOQA
from delayed_image.channel_spec import ChannelSpec  # NOQA
import ubelt as ub
ub.schedule_deprecation(
    'kwcoco', 'kwcoco.channel_spec', 'module', 'use delayed_image.channel_spec instead',
    deprecate='0.8.4', error='0.9.0', remove='0.10.0')
