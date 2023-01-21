"""
The ChannelSpec and FusedChannelSpec represent a set of channels or bands in an
image. This could be as simple as ``red|green|blue``, or more complex like:
``red|green|blue|nir|swir16|swir22``.

This functionality has been moved to "delayed_image".
"""
from delayed_image.channel_spec import FusedChannelSpec
from delayed_image.channel_spec import ChannelSpec
