from common.type_traits import is_value
from common.node import Node


class GrammerParser(object):

    def top_down_parsing(self, tokens):
        """Grammer parsing using top-down recursive manner.

        Grammer:
            E --> E+T
                | E-T
                | T

            T --> T*P
                | T/P
                | P

            P --> F^F
                | F

            F --> (E)
                | num
        
        after removal of left recursion:
            E --> TA
            A --> +TA | -TA | epsilon
            T --> PB
            B --> *PB | /PB | epsilon
            P --> FC
            C --> ^F | epsilon
            F --> (E) | num

        EBNF:
            E --> T (+|- T)*
            T --> P (*|/ P)*
            P --> FC
            C --> ^F | epsilon
            F --> (E) | num


        input:
            tokens: a list a str represent segmented elements in an expression.
        return:
            root: root node of ast.
        """
        self.tokens_list = list(reversed(tokens))
        try:
            root = self.parse_E()
        except NotImplementedError:
            print('There is a syntax error: ')
            print(tokens)
            # exit(1)
            return None
        except AssertionError:
            print('Lack of right bracket: ')
            print(tokens)
            # exit(1)
            return None
        return root

    def parse_E(self):
        node = self.parse_T()
        while True:
            token = self.next_token()
            if token in ['+', '-']:
                right = self.parse_T()
                node = Node(token, node, right)
            else:
                self.return_back(token)
                break
        return node
    
    def parse_T(self):
        node = self.parse_P()
        while True:
            token = self.next_token()
            if token in ['*', '/']:
                right = self.parse_P()
                node = Node(token, node, right)
            else:
                self.return_back(token)
                break
        return node
    
    def parse_P(self):
        left = self.parse_F()
        op, right = self.parse_C()
        if right is None:
            return left
        return Node(op, left, right)

    def parse_C(self):
        token = self.next_token()
        if token == '^':
            right = self.parse_F()
            return token, right
        self.return_back(token)
        return None, None

    def parse_F(self):
        token = self.next_token()
        if token == '(':
            node = self.parse_E()
            token = self.next_token()
            assert(token == ')')
            return node
        elif is_value(token):
            return Node(token)
        raise NotImplementedError('Parser error!')

    def next_token(self):
        if self.tokens_list:
            token = self.tokens_list[-1]
            self.tokens_list.pop()
        else:
            token = None
        return token

    def return_back(self, token):
        self.tokens_list.append(token)

