import pytest

from kwcoco.util.util_json import find_json_unserializable


def test_find_json_unserializable_cycle_detection():
    data = []
    data.append(data)
    with pytest.raises(ValueError, match='Circular reference detected'):
        list(find_json_unserializable(data))


def test_find_json_unserializable_allows_shared_dag():
    shared = {'shared': 1}
    data = [shared, shared]
    parts = list(find_json_unserializable(data))
    assert parts == []
