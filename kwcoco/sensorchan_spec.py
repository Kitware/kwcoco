"""
This is an extension of :mod:`kwcoco.channel_spec`, which augments channel
information with an associated sensor attribute. Eventually, this will entirely
replace the channel spec.
"""

import ubelt as ub
import functools
import itertools as it

try:
    import lark
except ImportError:
    lark = None

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


class Fused:
    def __init__(self, chan):
        self.chan = chan

    def __repr__(self):
        return f'{"|".join(self.chan)}'
        # return f'Fused({"|".join(self.chan)})'

    def __str__(self):
        return f'{"|".join(self.chan)}'
        # return f'Fused({"|".join(self.chan)})'


class SensorChan:
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


class ConciseFused:
    def __init__(self, chan):
        import kwcoco
        self.chan = chan
        self.concise = kwcoco.FusedChannelSpec.coerce(chan).concise()

    def __repr__(self):
        return self.concise.spec

    def __str__(self):
        return self.concise.spec


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

