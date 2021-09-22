"""
The ChannelSpec has these simple rules:

.. code::

    * each 1D channel is a alphanumeric string.

    * The pipe ('|') separates aligned early fused stremas (non-communative)

    * The comma (',') separates late-fused streams, (happens after pipe operations, and is communative)

    * Certain common sets of early fused channels have codenames, for example:

        rgb = r|g|b
        rgba = r|g|b|a
        dxdy = dy|dy

For single arrays, the spec is always an early fused spec.

TODO:
    - [X] : normalize representations? e.g: rgb = r|g|b? - OPTIONAL
    - [X] : rename to BandsSpec or SensorSpec? - REJECTED
    - [ ] : allow bands to be coerced, i.e. rgb -> gray, or gray->rgb


TODO:
    - [ ]: Use FusedChannelSpec as a member of ChannelSpec
    - [ ]: Handle special slice suffix for length calculations
"""
import ubelt as ub
import six
import functools
import abc


class BaseChannelSpec(ub.NiceRepr):
    """
    Common code API between :class:`FusedChannelSpec` and :class:`ChannelSpec`

    TODO:
        - [ ] Keep working on this base spec and ensure the inheriting classes
              conform to it.
    """

    @property
    @abc.abstractmethod
    def spec(self):
        """
        The string encodeing of this spec

        Returns:
            str
        """
        ...

    @classmethod
    @abc.abstractmethod
    def coerce(cls, data):
        """
        Try and interpret the input data as some sort of spec

        Args:
            data (str | int | list | dict | BaseChannelSpec):
                any input data that is known to represent a spec

        Returns:
            BaseChannelSpec
        """
        ...

    @abc.abstractmethod
    def streams(self):
        """
        Breakup this spec into individual early-fused components

        Returns:
            List[FusedChannelSpec]
        """
        ...

    @abc.abstractmethod
    def normalize(self):
        """
        Expand all channel codes into their normalized long-form

        Returns:
            BaseChannelSpec
        """
        ...

    @abc.abstractmethod
    def intersection(self):
        ...

    @abc.abstractmethod
    def difference(self):
        ...

    def __nice__(self):
        return self.spec

    def __json__(self):
        return self.spec

    def __and__(self, other):
        # the parent implementation of this is backwards
        return self.intersection(other)


class FusedChannelSpec(BaseChannelSpec):
    """
    A specific type of channel spec with only one early fused stream.

    The channels in this stream are non-communative

    Behaves like a list of atomic-channel codes
    (which may represent more than 1 channel), normalized codes always
    represent exactly 1 channel.

    Note:
        This class name and API is in flux and subject to change.

    TODO:
        A special code indicating a name and some number of bands that that
        names contains, this would primarilly be used for large numbers of
        channels produced by a network. Like:

            resnet_d35d060_L5:512

            or

            resnet_d35d060_L5[:512]

        might refer to a very specific (hashed) set of resnet parameters
        with 512 bands

        maybe we can do something slicly like:

            resnet_d35d060_L5[A:B]
            resnet_d35d060_L5:A:B

        Do we want to "just store the code" and allow for parsing later?

        Or do we want to ensure the serialization is parsed before we
        construct the data structure?

    Example:
        >>> from kwcoco.channel_spec import *  # NOQA
        >>> import pickle
        >>> self = FusedChannelSpec.coerce(3)
        >>> recon = pickle.loads(pickle.dumps(self))
        >>> self = ChannelSpec.coerce('a|b,c|d')
        >>> recon = pickle.loads(pickle.dumps(self))
    """

    _alias_lut = {
        'rgb': 'r|g|b',
        'rgba': 'r|g|b|a',
        'dxdy': 'dx|dy',
        'fxfy': 'fx|fy',
    }

    _size_lut = {k: v.count('|') + 1 for k, v in _alias_lut.items()}

    def __init__(self, parsed, _is_normalized=False):
        self.parsed = parsed
        # denote if we are already normalized or not for speed.
        self._is_normalized = _is_normalized

    def __len__(self):
        import warnings
        if not self._is_normalized:
            warnings.warn(ub.paragraph(
                '''
                Length Definition for unormalized FusedChannelSpec is in flux.

                It is unclear if it should be the (1) number of atomic codes or
                (2) the expanded "numel", which is the number of "normalized"
                atomic codes. Currently it returns the number "unnormalized"
                atomic codes. Normalizing the FusedChannelSpec object or using
                "numel" will supress this warning.
                '''))
        return len(self.parsed)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(self.parsed[index])
        elif ub.iterable(index):
            return self.__class__(list(ub.take(self.parsed, index)))
        else:
            return self.parsed[index]

    @classmethod
    def concat(cls, items):
        combined = list(ub.flatten(item.parsed for item in items))
        self = cls(combined)
        return self

    @ub.memoize_property
    def spec(self):
        return '|'.join(self.parsed)

    @ub.memoize
    def unique(self):
        return set(self.parsed)

    @classmethod
    def parse(cls, spec):
        self = cls(spec.split('|'))
        return self

    @classmethod
    def coerce(cls, data):
        """
        Example:
            >>> from kwcoco.channel_spec import *  # NOQA
            >>> FusedChannelSpec.coerce(['a', 'b', 'c'])
            >>> FusedChannelSpec.coerce('a|b|c')
            >>> FusedChannelSpec.coerce(3)
            >>> FusedChannelSpec.coerce(FusedChannelSpec(['a']))
        """
        if isinstance(data, list):
            self = cls(data)
        elif isinstance(data, str):
            self = cls.parse(data)
        elif isinstance(data, int):
            # we know the number of channels, but not their names
            self = cls(['u{}'.format(i) for i in range(data)])
        elif isinstance(data, cls):
            self = data
        elif isinstance(data, ChannelSpec):
            parsed = data.parse()
            if len(parsed) == 1:
                self = cls(ub.peek(parsed.values()))
            else:
                raise ValueError(
                    'Cannot coerce ChannelSpec to a FusedChannelSpec '
                    'when there are multiple streams')
        else:
            raise TypeError('unknown type {}'.format(type(data)))
        return self

    @ub.memoize_method
    def normalize(self):
        """
        Replace aliases with explicit single-band-per-code specs

        Returns:
            FusedChannelSpec : normalize spec

        Example:
            >>> from kwcoco.channel_spec import *  # NOQA
            >>> self = FusedChannelSpec.coerce('b1|b2|b3|rgb')
            >>> normed = self.normalize()
            >>> print('self   = {}'.format(ub.repr2(self, nl=1)))
            >>> print('normed = {}'.format(ub.repr2(normed, nl=1)))
            >>> self = FusedChannelSpec.coerce('B:1:11')
            >>> normed = self.normalize()
            >>> print('self   = {}'.format(ub.repr2(self, nl=1)))
            >>> print('normed = {}'.format(ub.repr2(normed, nl=1)))
        """
        if self._is_normalized:
            return self

        norm_parsed = []
        for v in self.parsed:
            if v in self._alias_lut:
                norm_parsed.extend(self._alias_lut.get(v).split('|'))
            else:
                if ':' in v:
                    root, *slice_args = v.split(':')
                    if len(slice_args) == 1:
                        start = 0
                        stop, = map(int, slice_args)
                        step = 1
                    elif len(slice_args) == 2:
                        start, stop = map(int, slice_args)
                        step = 1
                    else:
                        raise NotImplementedError
                    for idx in range(start, stop, step):
                        norm_parsed.append('{}.{}'.format(root, idx))
                else:
                    norm_parsed.append(v)
        # self._alias_lut.get(v, v).split('|')
        # norm_parsed = list(ub.flatten(
        #     for v in self.parsed))
        normed = FusedChannelSpec(norm_parsed, _is_normalized=True)
        return normed

    def numel(self):
        """
        Total number of channels in this spec
        """
        if self._is_normalized:
            return len(self.parsed)
        else:
            return sum(self.sizes())

    def sizes(self):
        """
        Returns a list indicating the size of each atomic code

        Returns:
            List[int]

        Example:
            >>> from kwcoco.channel_spec import *  # NOQA
            >>> self = FusedChannelSpec.coerce('b1|Z:3|b2|b3|rgb')
            >>> self.sizes()
            [1, 3, 1, 1, 3]
        """
        if self._is_normalized:
            return [1] * len(self.parsed)
        size_list = []
        for v in self.parsed:
            if v in self._alias_lut:
                num = self._alias_lut.get(v).count('|') + 1
            else:
                if ':' in v:
                    root, *slice_args = v.split(':')
                    if len(slice_args) == 1:
                        start = 0
                        stop, = map(int, slice_args)
                        step = 1
                    elif len(slice_args) == 2:
                        start, stop = map(int, slice_args)
                        step = 1
                    else:
                        raise NotImplementedError
                    num = len(range(start, stop, step))
                else:
                    num = 1
            size_list.append(num)
        return size_list

    def __contains__(self, key):
        """
        Example:
            >>> FCS = FusedChannelSpec.coerce
            >>> 'disparity' in FCS('rgb|disparity|flowx|flowy')
            True
            >>> 'gray' in FCS('rgb|disparity|flowx|flowy')
            False
        """
        return key in self.unique()

    # def can_coerce(self, other):
    #     # return if we can coerce this band repr to another, like
    #     # gray to rgb or rgb to gray

    def code_list(self):
        """
        Return the expanded code list
        """
        return self.parsed

    # @ub.memoize_property
    # def code_oset(self):
    #     return ub.oset(self.normalize().parsed)

    @ub.memoize_method
    def as_list(self):
        return self.normalize().parsed

    @ub.memoize_method
    def as_oset(self):
        return ub.oset(self.normalize().parsed)

    @ub.memoize_method
    def as_set(self):
        return set(self.normalize().parsed)

    def __set__(self):
        return self.as_set()

    def difference(self, other):
        """
        Set difference

        Example:
            >>> FCS = FusedChannelSpec.coerce
            >>> self = FCS('rgb|disparity|flowx|flowy')
            >>> other = FCS('r|b')
            >>> self.difference(other)
            >>> other = FCS('flowx')
            >>> self.difference(other)
        """
        other_norm = ub.oset(other.normalize().parsed)
        self_norm = ub.oset(self.normalize().parsed)
        new_parsed = list(self_norm - other_norm)
        new = self.__class__(new_parsed, _is_normalized=True)
        return new

    def intersection(self, other):
        """
        Example:
            >>> FCS = FusedChannelSpec.coerce
            >>> self = FCS('rgb|disparity|flowx|flowy')
            >>> other = FCS('r|b|XX')
            >>> self.intersection(other)
        """
        other_norm = ub.oset(other.normalize().parsed)
        self_norm = ub.oset(self.normalize().parsed)
        new_parsed = list(self_norm & other_norm)
        new = self.__class__(new_parsed, _is_normalized=True)
        return new

    def component_indices(self, axis=2):
        """
        Look up component indices within this stream

        Example:
            >>> FCS = FusedChannelSpec.coerce
            >>> self = FCS('disparity|rgb|flowx|flowy')
            >>> component_indices = self.component_indices()
            >>> print('component_indices = {}'.format(ub.repr2(component_indices, nl=1)))
            component_indices = {
                'disparity': (slice(...), slice(...), slice(0, 1, None)),
                'flowx': (slice(...), slice(...), slice(4, 5, None)),
                'flowy': (slice(...), slice(...), slice(5, 6, None)),
                'rgb': (slice(...), slice(...), slice(1, 4, None)),
            }
        """
        component_indices = dict()
        idx1 = 0
        for part in self.parsed:
            size = self._size_lut.get(part, 1)
            idx2 = idx1 + size
            index = tuple([slice(None)] * axis + [slice(idx1, idx2)])
            idx1 = idx2
            component_indices[part] = index
        return component_indices

    def streams(self):
        """
        Idempotence with :func:`ChannelSpec.streams`
        """
        return [self]

    def fuse(self):
        """
        Idempotence with :func:`ChannelSpec.streams`
        """
        return self


class ChannelSpec(BaseChannelSpec):
    """
    Parse and extract information about network input channel specs for
    early or late fusion networks.

    Behaves like a dictionary of FusedChannelSpec objects

    TODO:
        - [ ] Rename to something that indicates this is a collection of
            FusedChannelSpec? MultiChannelSpec?

    Note:
        This class name and API is in flux and subject to change.

    Note:
        The pipe ('|') character represents an early-fused input stream, and
        order matters (it is non-communative).

        The comma (',') character separates different inputs streams/branches
        for a multi-stream/branch network which will be lated fused. Order does
        not matter

    Example:
        >>> from kwcoco.channel_spec import *  # NOQA
        >>> # Integer spec
        >>> ChannelSpec.coerce(3)
        <ChannelSpec(u0|u1|u2) ...>

        >>> # single mode spec
        >>> ChannelSpec.coerce('rgb')
        <ChannelSpec(rgb) ...>

        >>> # early fused input spec
        >>> ChannelSpec.coerce('rgb|disprity')
        <ChannelSpec(rgb|disprity) ...>

        >>> # late fused input spec
        >>> ChannelSpec.coerce('rgb,disprity')
        <ChannelSpec(rgb,disprity) ...>

        >>> # early and late fused input spec
        >>> ChannelSpec.coerce('rgb|ir,disprity')
        <ChannelSpec(rgb|ir,disprity) ...>

    Example:
        >>> self = ChannelSpec('gray')
        >>> print('self.info = {}'.format(ub.repr2(self.info, nl=1)))
        >>> self = ChannelSpec('rgb')
        >>> print('self.info = {}'.format(ub.repr2(self.info, nl=1)))
        >>> self = ChannelSpec('rgb|disparity')
        >>> print('self.info = {}'.format(ub.repr2(self.info, nl=1)))
        >>> self = ChannelSpec('rgb|disparity,disparity')
        >>> print('self.info = {}'.format(ub.repr2(self.info, nl=1)))
        >>> self = ChannelSpec('rgb,disparity,flowx|flowy')
        >>> print('self.info = {}'.format(ub.repr2(self.info, nl=1)))

    Example:
        >>> specs = [
        >>>     'rgb',              # and rgb input
        >>>     'rgb|disprity',     # rgb early fused with disparity
        >>>     'rgb,disprity',     # rgb early late with disparity
        >>>     'rgb|ir,disprity',  # rgb early fused with ir and late fused with disparity
        >>>     3,                  # 3 unknown channels
        >>> ]
        >>> for spec in specs:
        >>>     print('=======================')
        >>>     print('spec = {!r}'.format(spec))
        >>>     #
        >>>     self = ChannelSpec.coerce(spec)
        >>>     print('self = {!r}'.format(self))
        >>>     sizes = self.sizes()
        >>>     print('sizes = {!r}'.format(sizes))
        >>>     print('self.info = {}'.format(ub.repr2(self.info, nl=1)))
        >>>     #
        >>>     item = self._demo_item((1, 1), rng=0)
        >>>     inputs = self.encode(item)
        >>>     components = self.decode(inputs)
        >>>     input_shapes = ub.map_vals(lambda x: x.shape, inputs)
        >>>     component_shapes = ub.map_vals(lambda x: x.shape, components)
        >>>     print('item = {}'.format(ub.repr2(item, precision=1)))
        >>>     print('inputs = {}'.format(ub.repr2(inputs, precision=1)))
        >>>     print('input_shapes = {}'.format(ub.repr2(input_shapes)))
        >>>     print('components = {}'.format(ub.repr2(components, precision=1)))
        >>>     print('component_shapes = {}'.format(ub.repr2(component_shapes, nl=1)))

    """

    def __init__(self, spec, parsed=None):
        # TODO: allow integer specs
        self._spec = spec
        self._info = {
            'spec': spec,
            'parsed': parsed,
        }

    @property
    def spec(self):
        return self._spec

    def __contains__(self, key):
        """
        Example:
            >>> 'disparity' in ChannelSpec('rgb,disparity,flowx|flowy')
            True
            >>> 'gray' in ChannelSpec('rgb,disparity,flowx|flowy')
            False
        """
        return key in self.unique()

    @property
    def info(self):
        return ub.dict_union(self._info, {
            'unique': self.unique(),
            'normed': self.normalize(),
        })

    @classmethod
    def coerce(cls, data):
        """
        Attempt to interpret the data as a channel specification

        Returns:
            ChannelSpec

        Example:
            >>> from kwcoco.channel_spec import *  # NOQA
            >>> data = FusedChannelSpec.coerce(3)
            >>> assert ChannelSpec.coerce(data).spec == 'u0|u1|u2'
            >>> data = ChannelSpec.coerce(3)
            >>> assert data.spec == 'u0|u1|u2'
            >>> assert ChannelSpec.coerce(data).spec == 'u0|u1|u2'
            >>> data = ChannelSpec.coerce('u:3')
            >>> assert data.normalize().spec == 'u.0|u.1|u.2'
        """
        if isinstance(data, cls):
            self = data
            return self
        elif isinstance(data, FusedChannelSpec):
            spec = data.spec
            parsed = {spec: data}
            self = cls(spec, parsed)
            return self
        else:
            if isinstance(data, int):
                # we know the number of channels, but not their names
                spec = '|'.join(['u{}'.format(i) for i in range(data)])
            elif isinstance(data, six.string_types):
                spec = data
            else:
                raise TypeError(type(data))

            self = cls(spec)
            return self

    def parse(self):
        """
        Build internal representation

        Example:
            >>> from kwcoco.channel_spec import *  # NOQA
            >>> self = ChannelSpec('b1|b2|b3|rgb,B:3')
            >>> print(self.parse())
            >>> print(self.normalize().parse())
        """
        if self._info.get('parsed', None) is None:
            # commas break inputs into multiple streams
            stream_specs = self.spec.split(',')
            # parsed = {ss: ss.split('|') for ss in stream_specs}
            parsed = {ss: FusedChannelSpec(ss.split('|'))
                      for ss in stream_specs}
            self._info['parsed'] = parsed
        return self._info['parsed']

    def normalize(self):
        """
        Replace aliases with explicit single-band-per-code specs

        Returns:
            ChannelSpec : normalized spec

        Example:
            >>> self = ChannelSpec('b1|b2,b3|rgb,B:3')
            >>> normed = self.normalize()
            >>> print('self   = {}'.format(self))
            >>> print('normed = {}'.format(normed))
            self   = <ChannelSpec(b1|b2,b3|rgb,B:3)>
            normed = <ChannelSpec(b1|b2,b3|r|g|b,B.0|B.1|B.2)>
        """
        new_parsed = {}
        for k1, v1 in self.parse().items():
            norm_vals = v1.normalize()
            norm_key = norm_vals.spec
            new_parsed[norm_key] = norm_vals
        new_spec = ','.join(list(new_parsed.keys()))
        normed = ChannelSpec(new_spec, parsed=new_parsed)
        return normed

    def keys(self):
        spec = self.spec
        stream_specs = spec.split(',')
        for spec in stream_specs:
            yield spec

    def values(self):
        return self.parse().values()

    def items(self):
        return self.parse().items()

    def fuse(self):
        """
        Fuse all parts into an early fused channel spec

        Returns:
            FusedChannelSpec

        Example:
            >>> from kwcoco.channel_spec import *  # NOQA
            >>> self = ChannelSpec.coerce('b1|b2,b3|rgb,B:3')
            >>> fused = self.fuse()
            >>> print('self  = {}'.format(self))
            >>> print('fused = {}'.format(fused))
            self  = <ChannelSpec(b1|b2,b3|rgb,B:3)>
            fused = <FusedChannelSpec(b1|b2|b3|rgb|B:3)>
        """
        parts = self.streams()
        if len(parts) == 1:
            return parts[0]
        else:
            return FusedChannelSpec(list(ub.flatten([p.parsed for p in parts])))

    def streams(self):
        """
        Breaks this spec up into one spec for each early-fused input stream

        Example:
            self = ChannelSpec.coerce('r|g,B1|B2,fx|fy')
            list(map(len, self.streams()))
        """
        streams = [FusedChannelSpec.coerce(spec) for spec in self.keys()]
        return streams

    def code_list(self):
        parsed = self.parse()
        if len(parsed) > 1:
            raise Exception(
                'Can only work on single-streams. '
                'TODO make class for single streams')
        return ub.peek(parsed.values())

    def difference(self, other):
        """
        Set difference. Remove all instances of other channels from
        this set of channels.

        Example:
            >>> from kwcoco.channel_spec import *
            >>> self = ChannelSpec('rgb|disparity,flowx|r|flowy')
            >>> other = ChannelSpec('rgb')
            >>> print(self.difference(other))
            >>> other = ChannelSpec('flowx')
            >>> print(self.difference(other))
            <ChannelSpec(disparity,flowx|flowy)>
            <ChannelSpec(r|g|b|disparity,r|flowy)>
        """
        # assert len(list(other.keys())) == 1, 'can take diff with one stream'
        other_norm = ChannelSpec.coerce(other).fuse().normalize()
        # other_norm = ub.oset(ub.peek(other.normalize().values()))
        self_norm = self.normalize()

        new_streams = []
        for parts in self_norm.values():
            new_stream = parts.difference(other_norm)
            # new_parts = ub.oset(parts) - ub.oset(other_norm)
            # shrink the representation of a complex r|g|b to an alias if
            # possible.
            # TODO: make this more efficient
            # for alias, alias_spec in self._alias_lut.items():
            #     alias_parts = ub.oset(alias_spec.split('|'))
            #     index = subsequence_index(new_parts, alias_parts)
            #     if index is not None:
            #         oset_delitem(new_parts, index)
            #         oset_insert(new_parts, index.start, alias)
            # new_stream = '|'.join(new_parts)
            new_streams.append(new_stream)
        new_spec = ','.join([s.spec for s in new_streams])
        new = self.__class__(new_spec)
        return new

    def numel(self):
        """
        Total number of channels in this spec
        """
        return sum(self.sizes().values())

    def sizes(self):
        """
        Number of dimensions for each fused stream channel

        IE: The EARLY-FUSED channel sizes

        Example:
            >>> self = ChannelSpec('rgb|disparity,flowx|flowy,B:10')
            >>> self.sizes()
        """
        print('self = {!r}'.format(self))
        print('self._info = {}'.format(ub.repr2(self._info, nl=1)))
        sizes = {
            key: vals.numel()
            for key, vals in self.parse().items()
        }
        return sizes

    def unique(self, normalize=False):
        """
        Returns the unique channels that will need to be given or loaded
        """
        import warnings
        if normalize:
            warnings.warn(
                'FIXME: These kwargs are broken, but does anything use it?')
        if normalize:
            return set(ub.flatten(self.parse().values()))
        else:
            return set(ub.flatten(self.normalize().values()))

    def _item_shapes(self, dims):
        """
        Expected shape for an input item

        Args:
            dims (Tuple[int, int]): the spatial dimension

        Returns:
            Dict[int, tuple]
        """
        item_shapes = {}
        parsed = self.parse()
        fused_keys = list(self.keys())
        for fused_key in fused_keys:
            components = parsed[fused_key]
            for mode_key, c in zip(components.parsed, components.sizes()):
                shape = (c,) + tuple(dims)
                item_shapes[mode_key] = shape
        return item_shapes

    def _demo_item(self, dims=(4, 4), rng=None):
        """
        Create an input that satisfies this spec

        Returns:
            dict: an item like it might appear when its returned from the
                `__getitem__` method of a :class:`torch...Dataset`.

        Example:
            >>> dims = (1, 1)
            >>> ChannelSpec.coerce(3)._demo_item(dims, rng=0)
            >>> ChannelSpec.coerce('r|g|b|disaprity')._demo_item(dims, rng=0)
            >>> ChannelSpec.coerce('rgb|disaprity')._demo_item(dims, rng=0)
            >>> ChannelSpec.coerce('rgb,disaprity')._demo_item(dims, rng=0)
            >>> ChannelSpec.coerce('rgb')._demo_item(dims, rng=0)
            >>> ChannelSpec.coerce('gray')._demo_item(dims, rng=0)
        """
        import kwarray
        rng = kwarray.ensure_rng(rng)
        item_shapes = self._item_shapes(dims)
        item = {
            key: rng.rand(*shape)
            for key, shape in item_shapes.items()
        }
        return item

    def encode(self, item, axis=0, mode=1):
        """
        Given a dictionary containing preloaded components of the network
        inputs, build a concatenated (fused) network representations of each
        input stream.

        Args:
            item (Dict[str, Tensor]): a batch item containing unfused parts.
                each key should be a single-stream (optionally early fused)
                channel key.
            axis (int, default=0): concatenation dimension

        Returns:
            Dict[str, Tensor]:
                mapping between input stream and its early fused tensor input.

        Example:
            >>> from kwcoco.channel_spec import *  # NOQA
            >>> import numpy as np
            >>> dims = (4, 4)
            >>> item = {
            >>>     'rgb': np.random.rand(3, *dims),
            >>>     'disparity': np.random.rand(1, *dims),
            >>>     'flowx': np.random.rand(1, *dims),
            >>>     'flowy': np.random.rand(1, *dims),
            >>> }
            >>> # Complex Case
            >>> self = ChannelSpec('rgb,disparity,rgb|disparity|flowx|flowy,flowx|flowy')
            >>> fused = self.encode(item)
            >>> input_shapes = ub.map_vals(lambda x: x.shape, fused)
            >>> print('input_shapes = {}'.format(ub.repr2(input_shapes, nl=1)))
            >>> # Simpler case
            >>> self = ChannelSpec('rgb|disparity')
            >>> fused = self.encode(item)
            >>> input_shapes = ub.map_vals(lambda x: x.shape, fused)
            >>> print('input_shapes = {}'.format(ub.repr2(input_shapes, nl=1)))

        Example:
            >>> # Case where we have to break up early fused data
            >>> import numpy as np
            >>> dims = (40, 40)
            >>> item = {
            >>>     'rgb|disparity': np.random.rand(4, *dims),
            >>>     'flowx': np.random.rand(1, *dims),
            >>>     'flowy': np.random.rand(1, *dims),
            >>> }
            >>> # Complex Case
            >>> self = ChannelSpec('rgb,disparity,rgb|disparity,rgb|disparity|flowx|flowy,flowx|flowy,flowx,disparity')
            >>> inputs = self.encode(item)
            >>> input_shapes = ub.map_vals(lambda x: x.shape, inputs)
            >>> print('input_shapes = {}'.format(ub.repr2(input_shapes, nl=1)))

            >>> # xdoctest: +REQUIRES(--bench)
            >>> #self = ChannelSpec('rgb|disparity,flowx|flowy')
            >>> import timerit
            >>> ti = timerit.Timerit(100, bestof=10, verbose=2)
            >>> for timer in ti.reset('mode=simple'):
            >>>     with timer:
            >>>         inputs = self.encode(item, mode=0)
            >>> for timer in ti.reset('mode=minimize-concat'):
            >>>     with timer:
            >>>         inputs = self.encode(item, mode=1)

        Ignore:
            import xdev
            _ = xdev.profile_now(self.encode)(item, mode=1)
        """
        import kwarray
        if len(item) == 0:
            raise ValueError('Cannot encode empty item')
        _impl = kwarray.ArrayAPI.coerce(ub.peek(item.values()))

        parsed = self.parse()
        # unique = self.unique()

        # TODO: This can be made much more efficient by determining if the
        # channels item can be directly translated to the result inputs. We
        # probably don't need to do the full decoding each and every time.

        if mode == 1:
            # Slightly more complex implementation that attempts to minimize
            # concat operations.
            item_keys = tuple(sorted(item.keys()))
            parsed_items = tuple(sorted([(k, tuple(v.parsed))
                                         for k, v in parsed.items()]))
            new_fused_indices = _cached_single_fused_mapping(
                item_keys, parsed_items, axis=axis)

            fused = {}
            for key, idx_list in new_fused_indices.items():
                parts = [item[item_key][item_sl] for item_key, item_sl in idx_list]
                if len(parts) == 1:
                    fused[key] = parts[0]
                else:
                    fused[key] = _impl.cat(parts, axis=axis)
        elif mode == 0:
            # Simple implementation that always does the full break down of
            # item components.
            components = {}
            # Determine the layout of the channels in the input item
            key_specs = {key: ChannelSpec(key) for key in item.keys()}
            for key, spec in key_specs.items():
                decoded = spec.decode({key: item[key]}, axis=axis)
                for subkey, subval in decoded.items():
                    components[subkey] = subval

            fused = {}
            for key, parts in parsed.items():
                fused[key] = _impl.cat([components[k] for k in parts], axis=axis)
        else:
            raise KeyError(mode)

        return fused

    def decode(self, inputs, axis=1):
        """
        break an early fused item into its components

        Args:
            inputs (Dict[str, Tensor]): dictionary of components
            axis (int, default=1): channel dimension

        Example:
            >>> from kwcoco.channel_spec import *  # NOQA
            >>> import numpy as np
            >>> dims = (4, 4)
            >>> item_components = {
            >>>     'rgb': np.random.rand(3, *dims),
            >>>     'ir': np.random.rand(1, *dims),
            >>> }
            >>> self = ChannelSpec('rgb|ir')
            >>> item_encoded = self.encode(item_components)
            >>> batch = {k: np.concatenate([v[None, :], v[None, :]], axis=0)
            ...          for k, v in item_encoded.items()}
            >>> components = self.decode(batch)

        Example:
            >>> # xdoctest: +REQUIRES(module:netharn, module:torch)
            >>> import torch
            >>> import numpy as np
            >>> dims = (4, 4)
            >>> components = {
            >>>     'rgb': np.random.rand(3, *dims),
            >>>     'ir': np.random.rand(1, *dims),
            >>> }
            >>> components = ub.map_vals(torch.from_numpy, components)
            >>> self = ChannelSpec('rgb|ir')
            >>> encoded = self.encode(components)
            >>> from netharn.data import data_containers
            >>> item = {k: data_containers.ItemContainer(v, stack=True)
            >>>         for k, v in encoded.items()}
            >>> batch = data_containers.container_collate([item, item])
            >>> components = self.decode(batch)
        """
        parsed = self.parse()
        components = dict()
        for key, parts in parsed.items():
            idx1 = 0
            for part, size in zip(parts.parsed, parts.sizes()):
                # size = self._size_lut.get(part, 1)
                idx2 = idx1 + size
                fused = inputs[key]
                index = tuple([slice(None)] * axis + [slice(idx1, idx2)])
                component = fused[index]
                components[part] = component
                idx1 = idx2
        return components

    def component_indices(self, axis=2):
        """
        Look up component indices within fused streams

        Example:
            >>> dims = (4, 4)
            >>> inputs = ['flowx', 'flowy', 'disparity']
            >>> self = ChannelSpec('disparity,flowx|flowy')
            >>> component_indices = self.component_indices()
            >>> print('component_indices = {}'.format(ub.repr2(component_indices, nl=1)))
            component_indices = {
                'disparity': ('disparity', (slice(None, None, None), slice(None, None, None), slice(0, 1, None))),
                'flowx': ('flowx|flowy', (slice(None, None, None), slice(None, None, None), slice(0, 1, None))),
                'flowy': ('flowx|flowy', (slice(None, None, None), slice(None, None, None), slice(1, 2, None))),
            }

        """
        parsed = self.parse()
        component_indices = dict()
        for key, parts in parsed.items():
            idx1 = 0
            for part, size in zip(parts.parsed, parts.sizes()):
                idx2 = idx1 + size
                index = tuple([slice(None)] * axis + [slice(idx1, idx2)])
                idx1 = idx2
                component_indices[part] = (key, index)
        return component_indices


@functools.lru_cache(maxsize=None)
def _cached_single_fused_mapping(item_keys, parsed_items, axis=0):
    item_indices = {}
    for key in item_keys:
        key_idxs = _cached_single_stream_idxs(key, axis=axis)
        for subkey, subsl in key_idxs.items():
            item_indices[subkey] = subsl

    fused_indices = {}
    for key, parts in parsed_items:
        fused_indices[key] = [item_indices[k] for k in parts]

    new_fused_indices = {}
    for key, idx_list in fused_indices.items():
        # Determine which continguous slices can be merged into a
        # single slice
        prev_key = None
        prev_sl = None

        accepted = []
        accum = []
        for item_key, item_sl in idx_list:
            if prev_key == item_key:
                if prev_sl.stop == item_sl[-1].start and prev_sl.step == item_sl[-1].step:
                    accum.append((item_key, item_sl))
                    continue
            if accum:
                accepted.append(accum)
                accum = []
            prev_key = item_key
            prev_sl = item_sl[-1]
            accum.append((item_key, item_sl))
        if accum:
            accepted.append(accum)
            accum = []

        # Merge the accumulated contiguous slices
        new_idx_list = []
        for accum in accepted:
            if len(accum) > 1:
                item_key = accum[0][0]
                first = accum[0][1]
                last = accum[-1][1]
                new_sl = list(first)
                new_sl[-1] = slice(first[-1].start, last[-1].stop, last[-1].step)
                new_sl = tuple(new_sl)
                new_idx_list.append((item_key, new_sl))
            else:
                new_idx_list.append(accum[0])
        val = new_idx_list
        new_fused_indices[key] = val
    return new_fused_indices


@functools.lru_cache(maxsize=None)
def _cached_single_stream_idxs(key, axis=0):
    """
    Ignore:
        hack for speed

        axis = 0
        key = 'rgb|disparity'

        # xdoctest: +REQUIRES(--bench)
        import timerit
        ti = timerit.Timerit(100, bestof=10, verbose=2)
        for timer in ti.reset('time'):
            with timer:
                _cached_single_stream_idxs(key, axis=axis)
        for timer in ti.reset('time'):
            with timer:
                ChannelSpec(key).component_indices(axis=axis)
    """
    # concat operations.
    key_idxs = ChannelSpec(key).component_indices(axis=axis)
    return key_idxs


def subsequence_index(oset1, oset2):
    """
    Returns a slice into the first items indicating the position of
    the second items if they exist.

    This is a variant of the substring problem.

    Returns:
        None | slice

    Example:
        >>> oset1 = ub.oset([1, 2, 3, 4, 5, 6])
        >>> oset2 = ub.oset([2, 3, 4])
        >>> index = subsequence_index(oset1, oset2)
        >>> assert index

        >>> oset1 = ub.oset([1, 2, 3, 4, 5, 6])
        >>> oset2 = ub.oset([2, 4, 3])
        >>> index = subsequence_index(oset1, oset2)
        >>> assert not index
    """
    if len(oset2) == 0:
        base = 0
    else:
        item1 = oset2[0]
        try:
            base = oset1.index(item1)
        except (IndexError, KeyError):
            base = None

    index = None
    if base is not None:
        sl = slice(base, base + len(oset2))
        subset = oset1[sl]
        if subset == oset2:
            index = sl
    return index


def oset_insert(self, index, obj):
    """
    Ignore:
        self = ub.oset()
        oset_insert(self, 0, 'a')
        oset_insert(self, 0, 'b')
        oset_insert(self, 0, 'c')
        oset_insert(self, 1, 'd')
        oset_insert(self, 2, 'e')
        oset_insert(self, 0, 'f')
    """
    if obj not in self:
        # Bump index of every item after the insert position
        for key in self.items[index:]:
            self.map[key] = self.map[key] + 1
        self.items.insert(index, obj)
        self.map[obj] = index


def oset_delitem(self, index):
    """
    for ubelt oset, todo contribute back to luminosoinsight

    >>> self = ub.oset([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> index = slice(3, 5)
    >>> oset_delitem(self, index)

    Ignore:
        self = ub.oset(['r', 'g', 'b', 'disparity'])
        index = slice(0, 3)
        oset_delitem(self, index)

    """
    if isinstance(index, slice) and index == ub.orderedset.SLICE_ALL:
        self.clear()
    else:
        if ub.orderedset.is_iterable(index):
            to_remove = [self.items[i] for i in index]
        elif isinstance(index, slice) or hasattr(index, "__index__"):
            to_remove = self.items[index]
        else:
            raise TypeError("Don't know how to index an OrderedSet by %r" % index)

        if isinstance(to_remove, list):
            # Modified version of discard slightly more efficient for multiple
            # items
            remove_idxs = sorted([self.map[key] for key in to_remove], reverse=True)

            for key in to_remove:
                del self.map[key]

            for idx in remove_idxs:
                del self.items[idx]

            for k, v in self.map.items():
                # I think there is a more efficient way to do this?
                num_after = sum(v >= i for i in remove_idxs)
                if num_after:
                    self.map[k] = v - num_after
        else:
            self.discard(to_remove)
