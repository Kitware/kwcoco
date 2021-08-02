"""
Testing:
    https://documen.tician.de/pymbolic/

    https://github.com/inducer/pymbolic/issues/45

Notes:

    https://github.com/inducer/pymbolic/issues/45

    The idea is to represent some operation (or computation) tree.

    +- Crop
    |
    +- Cat
       |
       +- Warp
       |  |
       |  +- Cat
       |     |
       |     +- Crop
       |     |  |
       |     |  +- Array
       |     |
       |     +- Warp
       |        |
       |        +- Array
       |
       +- Warp
           |
           +- Load

    C(A(W(A(C(L), W(L))), W(L)))


    A simplified version of this tree

    +- Cat
       |
       +- Warp
       |  |
       |  +- Crop
       |     |
       |      +- Array
       |
       +- Warp
       |  |
       |  +- Crop
       |     |
       |     + Array
       |
       +- Warp
           |
           +- Crop
              |
              +- Load

    Let C = crop
    Let W = warp
    Let A = warp
    Let L = load or data source

    A(W(C(L)), W(C(L)), W(C(L)))

    C(A(W(A(C(L), W(L))), W(L)))


    operations should be able to be optimized. We can:

        * Combine neighboring warp operations

        * Move all cats towards the top (associative)

        * Move all crops towards the leafs (not-communative, but we can
          compute how a crop changes as it move through a warp )

        * Data sources should always be leafs

          C * W * S = W * C' * S

          where C' is the adjusted crop in the warped space.
          Not sure how to represent it in this notation yet.
          Maybe:

              C'.corners = C.corners.warp(W.inv()) ?

    Idealized Rules: (none of these are impemented)

        * If the child of a crop is a warp, compute the equivalent warp with a
          crop child.

        * If the child of an op is a cat, for every child duplicate it and
          apply the op, these are the args for a new top-level cat.


    Goal:
        * The finalize implementation shouldn't need to pass operation
          information from layer to layer. Instead it should execute eagerly at
          each node.

        * The way this is mitigated is instead of optimizing the operation tree
          at finalize time, we have a explicit method to optimize the tree.
          Then calling finalize on this optimized tree guarentees the minimal
          computation.

TODO:
    - [ ] Algorithm to optimize an arbitrary tree

         https://www.researchgate.net/publication/314641363_A_new_algorithm_of_symbolic_expression_simplification_based_on_right-threaded_binary_tree/
"""
import pymbolic as pmbl
from pymbolic.mapper import IdentityMapper
from pymbolic.primitives import Expression
import numpy as np


class AutoInspectable(object):
    """
    Helper to provide automatic defaults for pymbolic expressions
    """

    def init_arg_names(self):
        return tuple(self._initkw().keys())

    def __getinitargs__(self):
        return tuple(self._initkw().values())

    def _initkw(self):
        import inspect
        from collections import OrderedDict
        sig = inspect.signature(self.__class__)
        initkw = OrderedDict()
        for name, info in sig.parameters.items():
            if not hasattr(self, name):
                raise NotImplementedError((
                    'Unable to introspect init args because the class '
                    'did not have attributes with the same names as the '
                    'constructor arguments'))
            initkw[name] = getattr(self, name)
        return initkw


class AutoExpression(AutoInspectable, Expression):
    pass


class Warp(AutoExpression):
    def __init__(self, sub_data, transform):
        self.sub_data = sub_data
        self.transform = transform

    mapper_method = "map_warp"


class ChanCat(AutoExpression):
    def __init__(self, components):
        self.components = components

    mapper_method = "map_chancat"


class RawImage(AutoExpression):
    def __init__(self, data):
        self.data = data

    mapper_method = "map_raw"


class WarpFusionMapper(IdentityMapper):
    def map_warp(self, expr):
        if isinstance(expr.sub_data, Warp):
            # Fuse neighboring warps
            t1 = expr.transform
            t2 = expr.sub_data.transform
            new_tf = t1 @ t2
            new_subdata = self.rec(expr.sub_data.sub_data)
            new = Warp(new_subdata, new_tf)
            return new
        elif isinstance(expr.sub_data, ChanCat):
            # A warp followed by a ChanCat becomes a ChanCat followed by that
            # warp
            tf = expr.transform
            new_components = []
            for comp in expr.sub_data.components:
                new_comp = Warp(comp, tf)
                new_components.append(new_comp)
            new = ChanCat(new_components)
            new = self.rec(new)
            return new
        else:
            return expr

    def map_chancat(self, expr):
        # ChanCat is associative
        new_components = []

        def _flatten(comps):
            for c in comps:
                if isinstance(c, ChanCat):
                    yield from _flatten(c.components)
                else:
                    yield c
        new_components = [self.rec(c) for c in _flatten(expr.components)]
        new = ChanCat(new_components)
        return new

    def map_raw(self, expr):
        return expr


class Transform:
    # temporary transform class for easier to read outputs in POC
    def __init__(self, f):
        self.f = f

    def __matmul__(self, other):
        return Transform(self.f * other.f)

    def __str__(self):
        return 'Tranform({})'.format(self.f)

    def __repr__(self):
        return 'Tranform({})'.format(self.f)



def demo():
    raw1 = RawImage('image1')
    w1_a = Warp(raw1, Transform(2))
    w1_b = Warp(w1_a, Transform(3))

    raw2 = RawImage('image2')
    w2_a = Warp(raw2, Transform(5))
    w2_b = Warp(w2_a, Transform(7))

    raw3 = RawImage('image3')
    w3_a = Warp(raw3, Transform(11))
    w3_b = Warp(w3_a, Transform(13))

    cat1 = ChanCat([w1_b, w2_b])

    warp_cat = Warp(cat1, Transform(17))

    cat2 = ChanCat([warp_cat, w3_b])

    mapper = WarpFusionMapper()

    print('cat2    = {!r}'.format(cat2))
    result1 = mapper(cat2)
    print('result1 = {!r}'.format(result1))
    result2 = mapper(result1)
    print('result2 = {!r}'.format(result2))
    result3 = mapper(result2)
    print('result3 = {!r}'.format(result3))

    # from pymbolic.interop.ast import ASTToPymbolic

if __name__ == '__main__':
    """
    CommandLine:
        pip install pymbolic
        python ~/code/kwcoco/dev/symbolic_delayed.py
    """
    demo()
