"""
pip install lark
"""
import ubelt as ub
import lark


SENSOR_CHAN_GRAMMAR = ub.codeblock(
    '''
    // SENSOR_CHAN_GRAMMAR
    ?start: stream

    // An identifier can contain spaces
    IDEN: ("_"|LETTER) ("_"|" "|LETTER|DIGIT)*

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
    IDEN: ("_"|LETTER) ("_"|" "|LETTER|DIGIT)*

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
        return Fused(list(ub.flatten(items)))

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


def sensor_channel_lark():
    import ubelt as ub
    import lark_cython

    # https://github.com/lark-parser/lark/blob/master/docs/_static/lark_cheatsheet.pdf

    sensor_channel_parser_cython = lark.Lark(SENSOR_CHAN_GRAMMAR,  start='start', parser='lalr', _plugins=lark_cython.plugins)
    sensor_channel_parser_python = lark.Lark(SENSOR_CHAN_GRAMMAR,  start='start', parser='lalr')
    # sensor_channel_parser = lark.Lark(SENSOR_CHAN_GRAMMAR,  start='start', parser='earley')

    channel_only_parser_cython = lark.Lark(CHANNEL_ONLY_GRAMMAR,  start='start', parser='lalr', _plugins=lark_cython.plugins)
    channel_only_parser_python = lark.Lark(CHANNEL_ONLY_GRAMMAR,  start='start', parser='lalr')

    codes = [
        '(S2,L8):(R|G|B.0:3,a),bc'
        'R',
        'R|G',
        'R|G|mat.0',
        'R|G|mat.0:3',
        'R|G|mat:3',
        'R|G|mat:3,lwir|inv:3|nir',
        'R,G,B,d|e.3',
        'S2:(R,G,B,d|e.3)',
        'S2:(R,G,B,d|e.3)',
        'S2:a:3,d:3',
        'S2:(a:3,d:3)',
        'S2:a:3',
        '(S2,L8,WV):(a:3,b:3,c:3)',
        '(S2,L8,WV):(a:3,b:3,c:3),rgb',
        'blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3|invariants.0:8|forest|built_up|water',
        '(S2,L8):R|G|B,X|Y|Z,WV:pan',
        '(S2,L8):R|G|B,X|Y|Z,WV:(pan,depth)',
        '(S2,L8,WV):(a|b|c,d)',
        '(S2,L8,WV):(a:3|b:3|c:3,d:3)',
        'fsdf fds:Q,R,WV:(a:3|boo bar:3|c:3,d:3),B:3,S:S:3',
    ]

    # codes = [
    #     '(S2,L8):red|green|blue,WV:red|green|blue|depth',
    #     '(S2,L8):red|green|blue,S2:landcover,WV:(red|green|blue,depth)',
    #     'red|green|blue',
    #     '(S2,L8,WV):(r|g|b,nir|swir,inv:4),(S2,L8):mat.0:3,WV:depth',
    # ]

    for code in codes:
        print('-----')
        print('code = {!r}'.format(code))
        tree = sensor_channel_parser_cython.parse(code)
        # print(tree.pretty())
        transformed = NormalizeTransformer().transform(tree)
        print('transformed = {}'.format(ub.repr2(transformed, nl=1)))
        print('-----')

        import kwcoco

    # Check to see time difference between parser vs existing
    chan_only_codes = [
        'R|G|mat.0',
        'R|G|mat.0:3',
        'R|G|mat:3',
        'R|G|mat:3,lwir|inv:3|nir',
        'R,G,B,d|e.3',
        'blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3|invariants.0:8|forest|built_up|water',
        'R|G|B.0:9,nir|swir|mat.0|l3.2:4'
    ]

    # Lark is about 10x slower than existing code

    for code in chan_only_codes:
        print('-----')
        print('code = {!r}'.format(code))
        import timerit
        ti = timerit.Timerit(100, bestof=10, verbose=1, unit='us')

        for timer in ti.reset('kwcoco.ChannelSpec'):
            with timer:
                chan_spec = kwcoco.ChannelSpec.coerce(code).normalize()

        try:
            # Is this fused
            fchan_spec = kwcoco.FusedChannelSpec.coerce(code).normalize()
        except Exception:
            print('* not fused')
            fchan_spec = None
        else:
            for timer in ti.reset('kwcoco.FusedChannelSpec'):
                with timer:
                    kwcoco.FusedChannelSpec.coerce(code).normalize()

        for timer in ti.reset('lark sensor_channel_parser_python'):
            with timer:
                tree = sensor_channel_parser_python.parse(code)

        for timer in ti.reset('lark sensor_channel_parser_cython'):
            with timer:
                tree = sensor_channel_parser_cython.parse(code)

        for timer in ti.reset('lark channel_only_parser_python'):
            with timer:
                tree = channel_only_parser_python.parse(code)

        for timer in ti.reset('lark channel_only_parser_cython'):
            with timer:
                tree = channel_only_parser_cython.parse(code)

        transformed = NormalizeTransformer().transform(tree)
        print('transformed = {!r}'.format(transformed))
        print('chan_spec = {!r}'.format(chan_spec))
        print('fchan_spec = {!r}'.format(fchan_spec))
        print('code = {!r}'.format(code))
        print('-----')
    print('sensor_channel_parser_python = {!r}'.format(sensor_channel_parser_python))

    # transformed = NormalizeTransformer().transform(tree)
    # print('transformed = {!r}'.format(transformed))

    # tree = sensor_channel_parser.parse('S2:R|G|B')
    # transformed = NormalizeTransformer().transform(tree)
    # print('transformed = {!r}'.format(transformed))

    # 'S2:red:3'

    # print(sensor_channel_parser.parse('R|G|B').pretty())
    # print(sensor_channel_parser.parse('R|G,B').pretty())
    # print(sensor_channel_parser.parse('S2:R|G|B').pretty())
    # print(sensor_channel_parser.parse('(S2,L8):(R|G|B,a)').pretty())
    # print(sensor_channel_parser.parse('(S2,L8):R|G|B').pretty())
    # print(sensor_channel_parser.parse('(S2,L8):R|G|B,X|Y|Z,WV:pan').pretty())
    # print(sensor_channel_parser.parse('(S2,L8):R|G|B,X|Y|Z,WV:(pan,depth)').pretty())
    # print(sensor_channel_parser.parse('blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3|invariants.0:8|forest|built_up|water').pretty())


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/dev/symbolic_channels.py
    """
    sensor_channel_lark()
