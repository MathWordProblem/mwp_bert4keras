

class Node(object):

    def __init__(self, element=None, left=None, right=None):
        self.element = element
        self.left = left
        self.right = right
    
    def pre_order(self):
        if self.element is None:
            return []
        if self.left is None:
            return [self.element]
        return [self.element] + self.left.pre_order() + self.right.pre_order()
    
    def in_order(self):
        if self.element is None:
            return []
        if self.left is None:
            return [self.element]
        return ['('] + self.left.in_order() + [self.element] + self.right.in_order() + [')']
    
    def post_order(self):
        if self.element is None:
            return []
        if self.left is None:
            return [self.element]
        return  self.left.post_order() + self.right.post_order() + [self.element]

