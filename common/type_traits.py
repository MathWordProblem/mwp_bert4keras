from common.config import Config


def is_int(n):
    try:
        float_n = float(n)
        int_n = int(float_n)
    except ValueError:
        return False
    else:
        return float_n == int_n

def is_float(n):
    try:
        float_n = float(n)
    except ValueError:
        return False
    else:
        return True


def is_number(n):
    return is_int(n) or is_float(n)


def is_percentage(n):
    return n[-1] == '%' and is_number(n[:-1])

def is_ratio(n):
    index = n.find('/')
    if index == -1:
        return False
    return is_number(n[:index]) and is_number(n[index+1:])

def is_value(n):
    return is_number(n) or is_percentage(n) or is_ratio(n)


def is_operator(n):
    return n in Config.OPERATIONS

