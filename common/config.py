
class Math23KConfig(object):
    """Here is configurable parameters."""
    # Maximum of variable number. Variables in a problem range from n_0 to n_(MAX_VARIABLE_NUM-1)
    MAX_VARIABLE_NUM = 15

    # Only consider these constants.
    CONSTANTS = ['1', '3.14']

    # Operations used by problems.
    OPERATIONS = ['+', '-', '*', '/', '^']

    # Maximum sentence length.
    MAX_LENGTH = 200

    # Minimum of term frequency.
    MIN_TERM_FREQUENCY = 5

    """Here is parameters calculated automatically."""
    
    # Variables' string.
    VARIABLES = ['n{}'.format(i) for i in range(MAX_VARIABLE_NUM)]

    # All options.
    ALL_OPTIONS = ['<PAD>', '<SOS>'] + CONSTANTS + OPERATIONS + VARIABLES

    # Decoder output dimension in each step.
    OUTPUT_DIM = len(ALL_OPTIONS)


class APE210KConfig(object):
    """Here is configurable parameters."""
    # Maximum of variable number. Variables in a problem range from n_0 to n_(MAX_VARIABLE_NUM-1)
    MAX_VARIABLE_NUM = 17

    # Only consider these constants.
    CONSTANTS = ['1', '2', '3', '3.14', '4', '5', '6', '8', '10', '100']

    # Operations used by problems.
    OPERATIONS = ['+', '-', '*', '/', '^']

    # Maximum sentence length.
    MAX_LENGTH = 120

    # Minimum of term frequency.
    MIN_TERM_FREQUENCY = 12

    """Here is parameters calculated automatically."""
    
    # Variables' string.
    VARIABLES = ['n{}'.format(i) for i in range(MAX_VARIABLE_NUM)]

    # All options.
    ALL_OPTIONS = ['<PAD>', '<SOS>'] + CONSTANTS + OPERATIONS + VARIABLES

    # Decoder output dimension in each step.
    OUTPUT_DIM = len(ALL_OPTIONS)


Config = APE210KConfig
