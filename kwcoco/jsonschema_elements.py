r"""
Functional interface into defining jsonschema structures.

See mixin classes for details.

Example:
    >>> from kwcoco.jsonschema_elements import *  # NOQA
    >>> elem = SchemaElements()
    >>> for base in SchemaElements.__bases__:
    >>>     print('\n\n====\nbase = {!r}'.format(base))
    >>>     attrs = [key for key in dir(base) if not key.startswith('_')]
    >>>     for key in attrs:
    >>>         value = getattr(elem, key)
    >>>         print('{} = {}'.format(key, value))

"""
import ubelt as ub


class Element(dict):
    """
    Example:
        >>> from kwcoco.coco_schema import *  # NOQA
        >>> self = Element(base={'type': 'demo'}, options={'opt1', 'opt2'})
        >>> new = self(opt1=3)
        >>> print('self = {}'.format(ub.repr2(self, nl=1)))
        >>> print('new = {}'.format(ub.repr2(new, nl=1)))
        >>> print('new2 = {}'.format(ub.repr2(new(), nl=1)))
        >>> print('new3 = {}'.format(ub.repr2(new(title='myvar'), nl=1)))
        >>> print('new4 = {}'.format(ub.repr2(new(title='myvar')(examples=['']), nl=1)))
        >>> print('new5 = {}'.format(ub.repr2(new(badattr=True), nl=1)))
    """
    __generics__ = {
        'title': NotImplemented,
        'description': NotImplemented,
        'default': NotImplemented,
        'examples': NotImplemented
    }

    def __init__(self, base, options={}, _magic=None):
        self._base = base

        if isinstance(options, (set, list, tuple)):
            options = {k: None for k in options}

        self._options = ub.dict_union(options, self.__generics__)
        self._magic = _magic
        super().__init__(base)

    def __call__(self, *args, **kw):
        # newbase = base | (kw & options)  # imagine all the syntax
        if self._magic:
            kw = self._magic(kw)
        newbase = ub.dict_union(self._base, ub.dict_isect(kw, self._options))
        new = Element(newbase, self._options, self._magic)
        return new

    def validate(self, instance=ub.NoParam):
        import jsonschema
        from jsonschema.validators import validator_for
        if instance is ub.NoParam:
            cls = validator_for(self)
            cls.check_schema(self)
            return self
        else:
            return jsonschema.validate(instance, schema=self)


class ScalarElements(object):
    """
    Single-valued elements
    """

    @property
    def NULL(self):
        """
        https://json-schema.org/understanding-json-schema/reference/null.html
        """
        return Element(base={'type': 'null'}, options={})

    @property
    def BOOLEAN(self):
        """
        https://json-schema.org/understanding-json-schema/reference/null.html
        """
        return Element(base={'type': 'boolean'}, options={})

    @property
    def STRING(self):
        """
        https://json-schema.org/understanding-json-schema/reference/string.html
        """
        return Element(base={'type': 'string'}, options={
            'pattern': 'some regex',
            'enum': ['Street', 'Avenue', 'Boulevard'],
            })

    @property
    def NUMBER(self):
        """
        https://json-schema.org/understanding-json-schema/reference/numeric.html#number
        """
        return Element(base={'type': 'number'}, options={
            'minimum': 0,
            'maximum': 100,
            'exclusiveMaximum': True,
        })

    @property
    def INTEGER(self):
        """
        https://json-schema.org/understanding-json-schema/reference/numeric.html#integer
        """
        return Element(base={'type': 'integer'}, options={
                'minimum': 0,
                'maximum': 100,
                'exclusiveMaximum': True,
        })


class QuantifierElements(object):
    """
    Quantifier types

    https://json-schema.org/understanding-json-schema/reference/combining.html#allof

    Example:
        >>> from kwcoco.jsonschema_elements import *  # NOQA
        >>> elem.ANYOF(elem.STRING, elem.NUMBER).validate()
        >>> elem.ONEOF(elem.STRING, elem.NUMBER).validate()
        >>> elem.NOT(elem.NULL).validate()
        >>> elem.NOT(elem.ANY).validate()
        >>> elem.ANY.validate()
    """
    @property
    def ANY(self):
        return Element({})

    def ALLOF(self, *TYPES):
        return Element({'allOf': list(TYPES)})

    def ANYOF(self, *TYPES):
        return Element({'anyOf': list(TYPES)})

    def ONEOF(self, *TYPES):
        return Element({'oneOf': list(TYPES)})

    def NOT(self, TYPE):
        return Element({'not': TYPE})


class ContainerElements:
    """
    Types that contain other types

    Example:
        >>> from kwcoco.jsonschema_elements import *  # NOQA
        >>> print(elem.ARRAY().validate())
        >>> print(elem.OBJECT().validate())
        >>> print(elem.OBJECT().validate())
    """

    def ARRAY(self, TYPE={}, **kw):
        """
        https://json-schema.org/understanding-json-schema/reference/array.html

        Example:
            >>> from kwcoco.jsonschema_elements import *  # NOQA
            >>> ARRAY(numItems=3)
            >>> schema = ARRAY(minItems=3)
            >>> schema.validate()
        """
        def _magic(kw):
            numItems = kw.pop('numItems', None)
            if numItems is not None:
                kw.update({
                    'minItems': numItems,
                    'maxItems': numItems,
                })
            return kw
        self = Element(
                base={'type': 'array', 'items': TYPE},
                options={
                    'contains': {'type': 'number'},
                    'minItems': 2,
                    'maxItems': 3,
                    'uniqueItems': True,
                    'additionalItems': {'type': 'string'}
                    },
                _magic=_magic)
        return self(**_magic(kw))

    def OBJECT(self, PROPERTIES={}, **kw):
        """
        https://json-schema.org/understanding-json-schema/reference/object.html


        Example:
            >>> import jsonschema
            >>> schema = elem.OBJECT()
            >>> jsonschema.validate({}, schema)
            >>> #
            >>> import jsonschema
            >>> schema = elem.OBJECT({
            >>>     'key1': elem.ANY(),
            >>>     'key2': elem.ANY(),
            >>> }, required=['key1'])
            >>> jsonschema.validate({'key1': None}, schema)
            >>> #
            >>> import jsonschema
            >>> schema = elem.OBJECT({
            >>>     'key1': elem.OBJECT({'arr': elem.ARRAY()}),
            >>>     'key2': elem.ANY(),
            >>> }, required=['key1'], title='a title')
            >>> schema.validate()
            >>> print('schema = {}'.format(ub.repr2(schema, nl=-1)))
            >>> jsonschema.validate({'key1': {'arr': []}}, schema)
        """
        self = Element(
                base={'type': 'object', 'properties': PROPERTIES},
                options={
                    'additionalProperties': False,
                    'required': [],
                    'propertyNames': [],
                    'minProperties': 0,
                    'maxProperties': float('inf'),
                    'dependencies': {},
                    }
                )
        return self(**kw)


class SchemaElements(
            ScalarElements,
            QuantifierElements,
            ContainerElements,
        ):
    """
    Functional interface into defining jsonschema structures.

    See mixin classes for details.

    References:
        https://json-schema.org/understanding-json-schema/

    TODO:
        - [ ] Generics: title, description, default, examples

    CommandLine:
        xdoctest -m /home/joncrall/code/kwcoco/kwcoco/jsonschema_elements.py SchemaElements

    Example:
        >>> from kwcoco.jsonschema_elements import *  # NOQA
        >>> elem = SchemaElements()
        >>> elem.ARRAY(elem.ANY())
        >>> schema = OBJECT({
        >>>     'prop1': ARRAY(INTEGER, minItems=3),
        >>>     'prop2': ARRAY(STRING, numItems=2),
        >>>     'prop3': ARRAY(OBJECT({
        >>>         'subprob1': NUMBER,
        >>>         'subprob2': NUMBER,
        >>>     }))
        >>> })
        >>> print('schema = {}'.format(ub.repr2(schema, nl=-1)))
        >>> print('schema = {}'.format(ub.repr2(schema, nl=2)))

        >>> TYPE = elem.OBJECT({
        >>>     'p1': ANY,
        >>>     'p2': ANY,
        >>> }, required=['p1'])
        >>> import jsonschema
        >>> inst = {'p1': None}
        >>> jsonschema.validate(inst, schema=TYPE)
        >>> #jsonschema.validate({'p2': None}, schema=TYPE)
    """


if 0:
    """
    from kwcoco.jsonschema_elements import *  # NOQA

    elem = SchemaElements()
    attrs = [key for key in dir(SchemaElements) if not key.startswith('_')]

    print('elem = SchemaElements()')
    for key in attrs:
        print('{key} = elem.{key}'.format(key=key))
    """

elem = SchemaElements()

ALLOF = elem.ALLOF
ANY = elem.ANY
ANYOF = elem.ANYOF
ARRAY = elem.ARRAY
BOOLEAN = elem.BOOLEAN
INTEGER = elem.INTEGER
NOT = elem.NOT
NULL = elem.NULL
NUMBER = elem.NUMBER
OBJECT = elem.OBJECT
ONEOF = elem.ONEOF
STRING = elem.STRING


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/kwcoco/jsonschema_elements.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
