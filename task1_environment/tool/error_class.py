class Error(Exception):
    """Base class for exceptions in this module."""
    pass

# not in use
class PolicyNotCompatible(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
