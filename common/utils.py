from common.abstract_syntax_tree import AbstractSyntaxTree
from common.cleaner import Cleaner


cleaner = Cleaner()


def to_prefix(equation):
    equation = ' '.join(cleaner.segment_equation(equation))
    tree = AbstractSyntaxTree(equation)
    equation = '|'.join(tree.pre_order())
    return equation


def _to_infix(tokens):
    if tokens[0] not in ['+', '-', '*', '/', '^']:
        return tokens[0], 1
    op = tokens[0]
    left, left_bound = _to_infix(tokens[1:])
    right, right_bound = _to_infix(tokens[left_bound+1:])
    return '({}){}({})'.format(left, op, right), left_bound + right_bound + 1


def to_infix(equation):
    tokens = equation.split('|')
    equation, bound = _to_infix(tokens)
    assert(bound == len(tokens))
    return equation

