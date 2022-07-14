"""
This is an extension of :mod:`kwcoco.channel_spec`, which augments channel
information with an associated sensor attribute. Eventually, this will entirely
replace the channel spec.
"""

import ubelt as ub
import itertools as it
import functools


try:
    cache = functools.cache
except AttributeError:
    cache = ub.memoize

try:
    from lark import Transformer
except ImportError:
    class Transformer:
        pass

SENSOR_CHAN_GRAMMAR = ub.codeblock(
    '''
    // SENSOR_CHAN_GRAMMAR
    ?start: stream

    // An identifier can contain spaces
    IDEN: ("_"|"*"|LETTER) ("_"|" "|"-"|"*"|LETTER|DIGIT)*

    chan_single : IDEN
    chan_getitem : IDEN "." INT
    chan_getslice_0b : IDEN ":" INT
    chan_getslice_ab : IDEN "." INT ":" INT

    // A channel code can just be an ID, or it can have a getitem
    // style syntax with a scalar or slice as an argument
    chan_code : chan_single | chan_getslice_0b | chan_getslice_ab | chan_getitem

    // Fused channels are an ordered sequence of channel codes (without sensors)
    fused : chan_code ("|" chan_code)*

    // A channel only part can be a fused channel or a sequence
    channel_rhs : fused | fused_seq

    // Channels can be specified in a sequence but must contain parens
    fused_seq : "(" fused ("," fused)* ")"

    // Sensors can be specified in a sequence but must contain parens
    sensor_seq : "(" IDEN ("," IDEN)* "):"

    sensor_lhs : (IDEN ":") | (sensor_seq)

    sensor_chan : sensor_lhs channel_rhs

    nosensor_chan : channel_rhs

    stream_item : sensor_chan | nosensor_chan

    // A stream is an unordered sequence of fused channels, that can
    // optionally contain sensor specifications.

    stream : stream_item ("," stream_item)*

    %import common.DIGIT
    %import common.LETTER
    %import common.INT
    ''')


class SensorChanSpec(ub.NiceRepr):
    """
    The public facing API for the sensor / channel specification

    Example:
        >>> # xdoctest: +REQUIRES(module:lark)
        >>> from kwcoco.sensorchan_spec import SensorChanSpec
        >>> self = SensorChanSpec('(L8,S2):BGR,WV:BGR,S2:nir,L8:land.0:4')
        >>> s1 = self.normalize()
        >>> s2 = self.concise()
        >>> streams = self.streams()
        >>> print(s1)
        >>> print(s2)
        >>> print('streams = {}'.format(ub.repr2(streams, sv=1, nl=1)))
        <SensorChanSpec(L8:BGR,S2:BGR,WV:BGR,S2:nir,L8:land.0|land.1|land.2|land.3)>
        <SensorChanSpec((L8,S2,WV):BGR,L8:land:4,S2:nir)>
        streams = [
            <SensorChanSpec(L8:BGR)>,
            <SensorChanSpec(S2:BGR)>,
            <SensorChanSpec(WV:BGR)>,
            <SensorChanSpec(S2:nir)>,
            <SensorChanSpec(L8:land.0|land.1|land.2|land.3)>,
        ]


    Example:
        >>> # Check with generic sensors
        >>> # xdoctest: +REQUIRES(module:lark)
        >>> from kwcoco.sensorchan_spec import SensorChanSpec
        >>> import kwcoco
        >>> self = SensorChanSpec('(*):BGR,*:BGR,*:nir,*:land.0:4')
        >>> self.concise().normalize()
        >>> s1 = self.normalize()
        >>> s2 = self.concise()
        >>> print(s1)
        >>> print(s2)
        <SensorChanSpec(*:BGR,*:BGR,*:nir,*:land.0|land.1|land.2|land.3)>
        <SensorChanSpec((*,*):BGR,*:(nir,land:4))>
        >>> import kwcoco
        >>> c = kwcoco.ChannelSpec.coerce('BGR,BGR,nir,land.0:8')
        >>> c1 = c.normalize()
        >>> c2 = c.concise()
        >>> print(c1)
        >>> print(c2)
    """
    def __init__(self, spec: str):
        self.spec: str = spec

    def __nice__(self):
        return self.spec

    @classmethod
    def coerce(cls, data):
        """
        Attempt to interpret the data as a channel specification

        Returns:
            SensorChanSpec

        Example:
            >>> # xdoctest: +REQUIRES(module:lark)
            >>> from kwcoco.sensorchan_spec import *  # NOQA
            >>> from kwcoco.sensorchan_spec import SensorChanSpec
            >>> data = SensorChanSpec.coerce(3)
            >>> assert SensorChanSpec.coerce(data).normalize().spec == '*:u0|u1|u2'
            >>> data = SensorChanSpec.coerce(3)
            >>> assert data.spec == 'u0|u1|u2'
            >>> assert SensorChanSpec.coerce(data).spec == 'u0|u1|u2'
            >>> data = SensorChanSpec.coerce('u:3')
            >>> assert data.normalize().spec == '*:u.0|u.1|u.2'
        """
        import kwcoco
        if isinstance(data, cls):
            self = data
            return self
        elif isinstance(data, str):
            self = cls(data)
            return self
        elif isinstance(data, kwcoco.FusedChannelSpec):
            spec = data.spec
            self = cls(spec)
            return self
        elif isinstance(data, kwcoco.ChannelSpec):
            spec = data.spec
            self = cls(spec)
            return self
        else:
            chan = kwcoco.ChannelSpec.coerce(data)
            self = cls(chan.spec)
            return self

    def normalize(self):
        new_spec = normalize_sensor_chan(self.spec)
        new = self.__class__(new_spec)
        return new

    def concise(self):
        new_spec = concise_sensor_chan(self.spec)
        new = self.__class__(new_spec)
        return new

    def streams(self):
        parts = sensorchan_normalized_parts(self.spec)
        streams = [SensorChanSpec(str(part)) for part in parts]
        return streams


class SensorChanNode:
    """
    """
    def __init__(self, sensor, chan):
        self.sensor = sensor
        self.chan = chan

    @property
    def spec(self):
        return f"{self.sensor}:{self.chan}"

    def __repr__(self):
        return self.spec
        # return f'SensorChanNode({self.sensor}:{self.chan})'

    def __str__(self):
        return self.spec
        # return f'SensorChanNode({self.sensor}:{self.chan})'


class FusedChanNode:
    """
    Example:
        s = FusedChanNode('a|b|c.0|c.1|c.2')
        c = s.concise()
        print(s)
        print(c)
    """
    def __init__(self, chan):
        import kwcoco
        self.data = kwcoco.FusedChannelSpec.coerce(chan)

    @property
    def spec(self):
        return self.data.spec

    def concise(self):
        return self.__class__(self.data.concise())

    def __repr__(self):
        return self.data.spec

    def __str__(self):
        return self.data.spec


class SensorChanTransformer(Transformer):
    """
    Given a parsed tree for a sensor-chan spec, can transform it into useful
    forms.

    TODO:
        Make the classes that hold the underlying data more robust such that
        they either use the existing channel spec or entirely replace it.
        (probably the former). Also need to add either a FusedSensorChan node
        that is restircted to only a single sensor and group of fused channels.

    Ignore:
        cases = [
             'S1:b:3',
             'S1:b:3,S2:b:3',
             'S1:b:3,S2:(b.0,b.1,b.2)',
        ]
        basis = {
            'concise_channels': [0, 1],
            'concise_sensors': [0, 1],
        }
        for spec in cases:
            print('')
            print('=====')
            print('spec = {}'.format(ub.repr2(spec, nl=1)))
            print('-----')
            for kwargs in ub.named_product(basis):
                sensor_channel_parser = _global_sensor_chan_parser()
                tree = sensor_channel_parser.parse(spec)
                transformed = SensorChanTransformer(**kwargs).transform(tree)
                print('')
                print('kwargs = {}'.format(ub.repr2(kwargs, nl=0)))
                print(f'transformed={transformed}')
            print('')
            print('=====')

    """

    def __init__(self, concise_channels=1, concise_sensors=1):
        self.consise_channels = concise_channels
        self.concise_sensors = concise_sensors

    def chan_id(self, items):
        code, = items
        return code.value

    def chan_single(self, items):
        code, = items
        return [code.value]

    def chan_getitem(self, items):
        code, index = items
        return [f'{code}.{index.value}']

    def chan_getslice_0b(self, items):
        code, btok = items
        return ['{}.{}'.format(code, index) for index in range(int(btok.value))]

    def chan_getslice_ab(self, items):
        code, atok, btok = items
        return ['{}.{}'.format(code, index) for index in range(int(atok.value), int(btok.value))]

    def chan_code(self, items):
        return items[0]

    def sensor_seq(self, items):
        return [s.value for s in items]

    def fused_seq(self, items):
        s = list(items)
        return s

    def fused(self, items):
        ret = FusedChanNode(list(ub.flatten(items)))
        if self.consise_channels:
            ret = ret.concise()
        return ret

    def channel_rhs(self, items):
        flat = []
        for item in items:
            if ub.iterable(item):
                flat.extend(item)
            else:
                flat.append(item)
        return flat

    def sensor_lhs(self, items):
        flat = []
        for item in items:
            if ub.iterable(item):
                flat.extend(item)
            else:
                flat.append(item.value)
        return flat

    def nosensor_chan(self, items):
        item, = items
        return [SensorChanNode('*', c) for c in item]

    def sensor_chan(self, items):
        assert len(items) == 2
        lhs, rhs = items
        new = []
        for a, b in it.product(lhs, rhs):
            new.append(SensorChanNode(a, b))
        return new

    def stream_item(self, items):
        item, = items
        return item

    def stream(self, items):
        flat_items = list(ub.flatten(items))
        # TODO: can probably improve this
        if self.concise_sensors:
            flat_sensors = [str(f.sensor) for f in flat_items]
            flat_chans = [str(f.chan) for f in flat_items]
            chan_to_sensors = ub.group_items(flat_sensors, flat_chans)

            pass1_sensors = []
            pass1_chans = []
            for chan, sensors in chan_to_sensors.items():
                sense_part = ','.join(sorted(sensors))
                if len(sensors) > 1:
                    sense_part = '({})'.format(sense_part)
                pass1_sensors.append(sense_part)
                pass1_chans.append(str(chan))

            pass2_parts = []
            sensor_to_chan = ub.group_items(pass1_chans, pass1_sensors)
            for sensor, chans in sensor_to_chan.items():
                chan_part = ','.join(chans)
                if len(chans) > 1:
                    chan_part = '({})'.format(chan_part)
                pass2_parts.append('{}:{}'.format(sensor, chan_part))

            parts = pass2_parts
            parts = sorted(parts)
        else:
            parts = flat_items
        return parts


@cache
def _global_sensor_chan_parser():
    # https://github.com/lark-parser/lark/blob/master/docs/_static/lark_cheatsheet.pdf
    import lark
    try:
        import lark_cython
        sensor_channel_parser = lark.Lark(SENSOR_CHAN_GRAMMAR,  start='start', parser='lalr', _plugins=lark_cython.plugins)
    except ImportError:
        sensor_channel_parser = lark.Lark(SENSOR_CHAN_GRAMMAR,  start='start', parser='lalr')
    return sensor_channel_parser


@cache
def normalize_sensor_chan(spec):
    """
    Example:
        >>> # xdoctest: +REQUIRES(module:lark)
        >>> from kwcoco.sensorchan_spec import *  # NOQA
        >>> spec = 'L8:mat:4,L8:red,S2:red,S2:forest|brush,S2:mat.0|mat.1|mat.2|mat.3'
        >>> r1 = normalize_sensor_chan(spec)
        >>> spec = 'L8:r|g|b,L8:r|g|b'
        >>> r2 = normalize_sensor_chan(spec)
        >>> print(f'r1={r1}')
        >>> print(f'r2={r2}')
        r1=L8:mat.0|mat.1|mat.2|mat.3,L8:red,S2:red,S2:forest|brush,S2:mat.0|mat.1|mat.2|mat.3
        r2=L8:r|g|b,L8:r|g|b
    """
    sensor_channel_parser = _global_sensor_chan_parser()
    tree = sensor_channel_parser.parse(spec)
    transformed = SensorChanTransformer(concise_sensors=0, concise_channels=0).transform(tree)
    new_spec = ','.join([n.spec for n in transformed])
    return new_spec


@cache
def concise_sensor_chan(spec):
    """
    Example:
        >>> # xdoctest: +REQUIRES(module:lark)
        >>> from kwcoco.sensorchan_spec import *  # NOQA
        >>> spec = 'L8:mat.0|mat.1|mat.2|mat.3,L8:red,S2:red,S2:forest|brush,S2:mat.0|mat.1|mat.2|mat.3'
        >>> concise_spec = concise_sensor_chan(spec)
        >>> normed_spec = normalize_sensor_chan(concise_spec)
        >>> concise_spec2 = concise_sensor_chan(normed_spec)
        >>> assert concise_spec2 == concise_spec
        >>> print(concise_spec)
        (L8,S2):(mat:4,red),S2:forest|brush
    """
    sensor_channel_parser = _global_sensor_chan_parser()
    tree = sensor_channel_parser.parse(spec)
    transformed = SensorChanTransformer(concise_sensors=1, concise_channels=1).transform(tree)
    new_spec = ','.join([str(n) for n in transformed])
    return new_spec


# @cache
def sensorchan_concise_parts(spec):
    """
    Ignore:
        >>> # xdoctest: +REQUIRES(module:lark)
        >>> spec = 'L8:mat.0|mat.1|mat.2|mat.3,L8:red,(MODIS,S2):a|b|c,S2:red,S2:forest|brush|bare_ground,S2:mat.0|mat.1|mat.2|mat.3'
        >>> parts = sensorchan_concise_parts(spec)
    """
    sensor_channel_parser = _global_sensor_chan_parser()
    tree = sensor_channel_parser.parse(spec)
    transformed = SensorChanTransformer(concise_sensors=1, concise_channels=1).transform(tree)
    return transformed


def sensorchan_normalized_parts(spec):
    """
    Ignore:
        >>> # xdoctest: +REQUIRES(module:lark)
        >>> spec = 'L8:mat.0|mat.1|mat.2|mat.3,L8:red,(MODIS,S2):a|b|c,S2:red,S2:forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field,S2:mat.0|mat.1|mat.2|mat.3'
        >>> parts = sensorchan_normalized_parts(spec)
    """
    sensor_channel_parser = _global_sensor_chan_parser()
    tree = sensor_channel_parser.parse(spec)
    transformed = SensorChanTransformer(concise_sensors=0, concise_channels=0).transform(tree)
    return transformed
