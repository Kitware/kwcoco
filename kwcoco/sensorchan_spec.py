"""
This functionality has been moved to "delayed_image"
"""
from delayed_image.sensorchan_spec import SensorChanSpec, FusedSensorChanSpec
from delayed_image.sensorchan_spec import sensorchan_concise_parts
from delayed_image.sensorchan_spec import normalize_sensor_chan
from delayed_image.sensorchan_spec import concise_sensor_chan
from delayed_image.sensorchan_spec import sensorchan_normalized_parts
import ubelt as ub
ub.schedule_deprecation(
    'kwcoco', 'kwcoco.sensorchan_spec', 'module', 'use delayed_image.sensorchan_spec instead',
    deprecate='0.8.4', error='0.9.0', remove='0.10.0')
