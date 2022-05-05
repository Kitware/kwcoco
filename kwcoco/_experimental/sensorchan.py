"""
An experimental grammar for combined sensor / channels specifications
"""
import ubelt as ub
import lark
import functools


# TODO: Should remove the "-" from the spec, so we can eventually extend it

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

CHANNEL_ONLY_GRAMMAR = ub.codeblock(
    '''
    // CHANNEL_ONLY_GRAMMAR
    ?start: stream

    // An identifier can contain spaces
    IDEN: ("_"|LETTER) ("_"|" "|"-"|LETTER|DIGIT|"*")*

    chan_single : IDEN
    chan_getitem : IDEN "." INT
    chan_getslice_0b : IDEN ":" INT
    chan_getslice_ab : IDEN "." INT ":" INT

    // A channel code can just be an ID, or it can have a getitem
    // style syntax with a scalar or slice as an argument
    chan_code : chan_single | chan_getslice_0b | chan_getslice_ab | chan_getitem

    // Fused channels are an ordered sequence of channel codes (without sensors)
    fused : chan_code ("|" chan_code)*

    // Channels can be specified in a sequence but must contain parens
    fused_seq : "(" fused ("," fused)* ")"

    channel_rhs : fused | fused_seq

    stream : channel_rhs ("," channel_rhs)*

    %import common.DIGIT
    %import common.LETTER
    %import common.INT
    ''')


class Fused(ub.NiceRepr):
    def __init__(self, chan):
        self.chan = chan

    def __repr__(self):
        return f'{"|".join(self.chan)}'
        # return f'Fused({"|".join(self.chan)})'

    def __str__(self):
        return f'{"|".join(self.chan)}'
        # return f'Fused({"|".join(self.chan)})'


class SensorChan(ub.NiceRepr):
    def __init__(self, sensor, chan):
        self.sensor = sensor
        self.chan = chan

    def __repr__(self):
        return f"'{self.sensor}:{self.chan}'"
        # return f'SensorChan({self.sensor}:{self.chan})'

    def __str__(self):
        return f"{self.sensor}:{self.chan}"
        # return f'SensorChan({self.sensor}:{self.chan})'


class NormalizeTransformer(lark.Transformer):

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
        ret = Fused(list(ub.flatten(items)))
        print(f'{ret=}')
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
        return [SensorChan('*', c) for c in item]

    def sensor_chan(self, items):
        import itertools as it
        assert len(items) == 2
        lhs, rhs = items
        new = []
        for a, b in it.product(lhs, rhs):
            new.append(SensorChan(a, b))
        return new

    def stream_item(self, items):
        item, = items
        return item

    def stream(self, items):
        # return list(ub.flatten(items))
        return ','.join(list(map(str, ub.flatten(items))))


class ConciseFused(ub.NiceRepr):
    def __init__(self, chan):
        self.chan = chan
        import kwcoco
        self.concise = kwcoco.FusedChannelSpec.coerce(chan).concise()

    def __repr__(self):
        return self.concise.spec
        # return f'Fused({"|".join(self.chan)})'

    def __str__(self):
        return self.concise.spec
        # return f'Fused({"|".join(self.chan)})'


class ConciseTransformer(lark.Transformer):

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
        ret = ConciseFused(list(ub.flatten(items)))
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
        return [SensorChan('*', c) for c in item]

    def sensor_chan(self, items):
        import itertools as it
        assert len(items) == 2
        lhs, rhs = items
        new = []
        for a, b in it.product(lhs, rhs):
            new.append(SensorChan(a, b))
        return new

    def stream_item(self, items):
        item, = items
        return item

    def stream(self, items):
        flat_items = list(ub.flatten(items))
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
        concise = ','.join(sorted(parts))
        return concise


class ConcisePartsTransformer(ConciseTransformer):
    def stream(self, items):
        flat_items = list(ub.flatten(items))
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
        concise = sorted(parts)
        return concise


# @ub.memoize

@functools.cache
def _global_sensor_chan_parser():
    # https://github.com/lark-parser/lark/blob/master/docs/_static/lark_cheatsheet.pdf
    try:
        import lark_cython
        sensor_channel_parser = lark.Lark(SENSOR_CHAN_GRAMMAR,  start='start', parser='lalr', _plugins=lark_cython.plugins)
    except ImportError:
        sensor_channel_parser = lark.Lark(SENSOR_CHAN_GRAMMAR,  start='start', parser='lalr')
    return sensor_channel_parser


@functools.cache
def concise_sensor_chan(spec):
    """
    Ignore:
        spec = 'L8:matseg.0|matseg.1|matseg.2|matseg.3,L8:red,S2:red,S2:forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field,S2:matseg.0|matseg.1|matseg.2|matseg.3'
    """
    sensor_channel_parser = _global_sensor_chan_parser()
    tree = sensor_channel_parser.parse(spec)
    # transformed = NormalizeTransformer().transform(tree)
    transformed = ConciseTransformer().transform(tree)
    return transformed
    # print('transformed = {}'.format(ub.repr2(transformed, nl=1)))


@functools.cache
def sensorchan_parts(spec):
    """
    Ignore:
        spec = 'L8:matseg.0|matseg.1|matseg.2|matseg.3,L8:red,(PLANET,S2):a|b|c,S2:red,S2:forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field,S2:matseg.0|matseg.1|matseg.2|matseg.3'
        spec = concise_sensor_chan(spec)
        spec = '*:rgb'
    """
    sensor_channel_parser = _global_sensor_chan_parser()
    tree = sensor_channel_parser.parse(spec)
    transformed = ConcisePartsTransformer().transform(tree)
    return transformed
    # print('transformed = {}'.format(ub.repr2(transformed, nl=1)))
