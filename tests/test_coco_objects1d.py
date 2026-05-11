import pytest


def test_object_groups_private_lookup_is_explicitly_unsupported():
    from kwcoco.coco_objects1d import ObjectGroups

    groups = ObjectGroups([], dset=None)
    with pytest.raises(NotImplementedError, match='ObjectGroups._lookup'):
        groups._lookup('id')
