import re

from common.type_traits import is_value


class Cleaner(object):

    operators = ['+', '-', '*', '/', '(', ')', '^']

    def __init__(self):
        pass

    def _replace_power(self, text):
        text = text.replace('**', '^')
        text = text.replace('* *', '^')
        return text
    
    def _remove_plus_100_percentage(self, text):
        text = text.replace('*100%', '')
        return text

    def _replace_ratio_colon(self, text):
        text = text.replace(':', '/')
        return text
    
    def _merge_ratio(self, text):
        text = re.sub('(\d)\s*/\s*(\d+)', '\\1/\\2', text)
        return text

    def _remove_header(self, equation, header='x='):
        return equation.strip(header)

    def _remove_rear(self, equation, rear='.'):
        return equation.strip(rear)

    def _replate_square_brackets(self, equation):
        equation = equation.replace('[', '(')
        equation = equation.replace(']', ')')
        return equation

    def _split_english_value(self, text):
        """e.g. 128km --> 128 km"""
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        return text

    def _remove_redundant_brackets(self, text):
        """e.g. (5/16) --> 5/16"""
        text = re.sub(r'\((\d+/\d+)\)', r' \1 ', text)
        return text

    def segment_equation(self, equation, padding=True):
        if padding:
            for operator in self.operators:
                equation = equation.replace(operator, ' {} '.format(operator))
        equation = equation.split(' ')
        return [e for e in equation if e != '']

    def clean_problem_text(self, text, merge_ratio=False):
        text = self._replace_power(text)
        if merge_ratio:
            text = self._merge_ratio(text)
        text = self._split_english_value(text)
        text = self._remove_redundant_brackets(text)
        return text

    def clean_equation(self, equation):
        equation = self._remove_header(equation)
        equation = self._remove_rear(equation)
        equation = self._replace_power(equation)
        equation = self._remove_plus_100_percentage(equation)
        equation = self._replace_ratio_colon(equation)
        equation = self._replate_square_brackets(equation)
        equation = ' '.join(self.segment_equation(equation))
        return equation

    def add_hidden_multiply(self, tokens):
        """Add hidden multilication sign.

        For example:
            5(2+3) -->  5*(2+3)
        """
        new_tokens = []
        for token in tokens:
            if token == '(':
                if new_tokens and is_value(new_tokens[-1]):
                    new_tokens.append('*')
            elif is_value(token):
                if new_tokens and new_tokens[-1] == ')':
                    new_tokens.append('*')
            new_tokens.append(token)
        return new_tokens

    def remove_redundant_plus(self, tokens):
        """
        If plus sign is behind of '+' or '/' or '*' or '-' or '^', it will be ignored.

        """
        new_tokens = []
        for token in tokens:
            if token == '+':
                if not new_tokens or new_tokens[-1] == '(':
                    continue
                if new_tokens and new_tokens[-1] in ['+', '-', '*', '/', '^']:
                    continue
            new_tokens.append(token)
        return new_tokens
    
    def transfer_op_to_sigh(self, tokens):
        """Transfer minus/plus sign to negative/positive sign if the sign hasn't precceding number.
        If minus/plus sign is ahead of '(', a precceding zero will be padded.

        For example:
            6 + ( - 4 - 3 )  -->   6 + ( -4 - 3 )
            (-(3+4)+5) --> (0-(3+4)+5)
        """
        new_tokens = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in ['+', '-']:
                if not new_tokens or new_tokens[-1] == '(':
                    assert(i + 1 < len(tokens))
                    if tokens[i + 1] == '(':
                        # Padding zero before.
                        new_tokens.append('0')
                        new_tokens.append(token)
                    else:
                        # Combine with number.
                        new_tokens.append(token + tokens[i + 1])
                        i += 1
                else:
                    new_tokens.append(token)
            else:
                new_tokens.append(token)
            i += 1
        return new_tokens

    def remove_redundant_space(self, tokens):
        return [token for token in tokens if token != '']


if __name__ == '__main__':
    cleaner = Cleaner()
    print(cleaner._merge_ratio('已 知 ( 1 / ( m ) ) * ( 1 / 5 ) = 1 ． 如 果 m = 3 ， 那 么 n = 多 少'))
