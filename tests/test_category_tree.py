

def test_category_tree_contains():
    from kwcoco.category_tree import CategoryTree
    self = CategoryTree.demo('coco')
    for catname1 in self:
        catname2 = catname1 + 'does_not_exist'
        assert catname1 in self
        assert catname2 not in self

    from kwcoco.category_tree import CategoryTree
    self = CategoryTree.demo('btree2')
    for catname1 in self:
        catname2 = catname1 + 'does_not_exist'
        assert catname1 in self
        assert catname2 not in self


def test_category_tree_getitem():
    from kwcoco.category_tree import CategoryTree
    self = CategoryTree.demo('coco')
    for k in range(len(self)):
        self[k]

    import pytest
    with pytest.raises(IndexError):
        k = len(self) + 1
        self[k]

    with pytest.raises(TypeError):
        self[None]
