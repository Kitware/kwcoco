import pytest


def test_object_groups_private_lookup_is_explicitly_unsupported():
    from kwcoco.coco_objects1d import ObjectGroups

    groups = ObjectGroups([], dset=None)
    with pytest.raises(NotImplementedError, match='ObjectGroups._lookup'):
        groups._lookup('id')


def test_coco_image_view_slice():
    from kwcoco.coco_image import CocoImage
    from kwcoco.coco_objects1d import CocoImageView

    seen: list[int] = []
    objects = {gid: object.__new__(CocoImage) for gid in [10, 20, 30, 40]}

    def getter(gid: int) -> CocoImage:
        seen.append(gid)
        return objects[gid]

    view = CocoImageView(getter, [10, 20, 30, 40])
    assert view[1] is objects[20]
    assert view[1:3] == [objects[20], objects[30]]
    assert seen == [20, 20, 30]
