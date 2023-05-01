class TaskNotFoundException(Exception):
    """No available task to run
    
    Note:
        If the message needs to be translated. Consider using
        `ErrorMessagePromise`
    
    """

    def __init__(self, message):
        """The constructor for ErrorMessage

        Args:
            message (str): The message to be shown to the
                front-end user.

        """
        self.message = message

    def get(self):
        """The getter of ErrorMessage"""
        return self.message

    def __str__(self):
        """Strinify the object."""
        return self.message

