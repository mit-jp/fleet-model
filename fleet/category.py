
class Category:
    """Rudimentary ordered tree data structure for vehicle classes."""
    _parent = None
    _children = []
    _label = ''

    def __init__(self, label, children=dict(), parent=None):
        self._parent = parent
        self._label = label
        try:
            self._children = list([Category(k, v, self) for k, v in
                                   children.items()])
        except AttributeError:
            pass

    def __str__(self):
        return self._label

    def children(self):
        return list(map(str, self._children))

    def parent(self):
        return str(self._parent)

    def nodes(self):
        return sum([child.nodes() for child in self._children], [self._label])

    def leaves(self, root):
        if len(self._children) == 0:
            return self._label
        else:
            return sum([child.leaves() for child in self._children], [])

    def find(self, label):
        """Return the subtree with *label* at its head."""
        if label == self._label:
            return self
        for child in self._children:
            result = child.find(label)
            if result:
                return result
        return None
