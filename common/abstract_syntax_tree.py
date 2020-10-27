from common.grammer_parser import GrammerParser
from common.node import Node
from common.cleaner import Cleaner


class AbstractSyntaxTree(object):

    def __init__(self, segmented_equation):
        """equation text must be segmented. e.g, 2.75 - 1 ( 5/6 ) + 3 ( 1/4 ) - 2 ( 1/6 )"""
        self.cleaner = Cleaner()
        segmented_equation = self.cleaner.segment_equation(segmented_equation, padding=False)
        self.root = self._build_from_in_order_stream(segmented_equation)

    def _build_from_in_order_stream(self, tokens):
        # Add hidden * manually.
        tokens = self.cleaner.add_hidden_multiply(tokens)
        tokens = self.cleaner.remove_redundant_plus(tokens)
        tokens = self.cleaner.transfer_op_to_sigh(tokens)
        return GrammerParser().top_down_parsing(tokens)

    def pre_order(self):
        if not self.root:
            return []
        return self.root.pre_order()
    
    def in_order(self):
        if not self.root:
            return []
        return self.root.in_order()

    def post_order(self):
        if not self.root:
            return []
        return self.root.post_order()


if __name__ == '__main__':
    equation = '1 - 2 + 3'
    tree = AbstractSyntaxTree(equation)
    print(tree.pre_order())
