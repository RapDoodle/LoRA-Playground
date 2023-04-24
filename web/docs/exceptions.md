# Exception Handling

Mangee provides a unified way of handling exceptions.

## Introduction

The `core.exception` module contains two classes, `ErrorMessage` and `ErrorMessagePromise`. `ErrorMessage` inherits from Python's `Exception` class. It contains an error message that expected to be shown to the front end. `ErrorMessagePromise` works similarly, but instead of providing the message directly, it takes in the name of the string to cope with multi-language environments. It is a promise that the error message in the user's correct language will be provided to the user. The function `excpetion_handler` is a wrapper that encapsulates exceptions and aborts user requests whenever exceptions are raised.

## Usage

### ErrorMessage

This class is normally used in models and raised when errors occur(eg., validation errors are detected). For example,
```python
from core.exception import ErrorMessage

if not is_valid_username(username):
    raise ErrorMessage('Invalid username.')
```
Then wrap the resource endpoint with excpetion_handler
```python
from core.exception import excpetion_handler

@excpetion_handler
    def post(self):
        pass
```
Whenever the `ErrorMessage` is raised, the front end will receive the error message in JSON format with a status code of 400.
```json
{
    "error": "Invalid username."
}
```

### ErrorMessagePromise

This class works similarly to `ErrorMessage`, but the constructor takes in the name of the string instead of the message directly.
```python
from core.exception import ErrorMessagePromise

if not is_valid_username(username):
    raise ErrorMessagePromise('INVALID_USERNAME')
```
In the language resource file, it is defined that 
```xml
<string name="INVALID_USERNAME">Invalid username.</string>
```
Then wrap the resource endpoint with excpetion_handler
```python
from core.exception import excpetion_handler

@excpetion_handler
    def post(self):
        pass
```
Whenever the `ErrorMessagePromise` is raised, the front end will receive the error message in the user's preferred language in JSON format with a status code of 400.
```json
{
    "error": "Invalid username."
}
```

### Handling Unknown Exceptions

For exceptions other than `ErrorMessage` and `ErrorMessagePromise`, when the resource endpoint was wrapped with `exception_handler`, a message indicating internal error will be returned. By default, it returns the string named `INTERNAL_ERROR` text from `en-US.xml` when the application's default language is set to `en-US`.
```json
{
    "error": "Internal error."
}
```

## Author
- Bowen WU (@RapDoodle)