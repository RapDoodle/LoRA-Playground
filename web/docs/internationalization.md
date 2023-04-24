# Internationalization

The language module provides an easy way to extend your flask application to support multiple languages. In the startup project, we've implemented several examples in using the internationalization module.

## Getting Started
In this example, we will demonstrate the internationalization of `en-US` and `zh-CN`, which is the Chinese used in mainland China. We also implemented `zh-HK` in the example code, which is traditional Chinese used in Hong Kong SAR.

1. Prepare a resource file in the `langs` folder, with the name of the desired language. For American English, it could be `en-US.xml`. For British English, it could be `en-UK.xml`. But again, this is up to you. We will use `en-US.xml` and `zh-CN.xml` in this example.
1. Add the following code in `en-US.xml`
    ```xml
    <resources>
        <string name="INTERNAL_ERROR">Server internal error.</string>
        <string name="NOT_FOUND">Not found.</string>
        <string name="INVALID_USERNAME">Invalid username.</string>
        <string name="INVALID_PASSWORD">Invalid password.</string>
        <string name="ADDED">"%(obj_name)s" has been created successfully.</string>
        <string name="DELETED">"%(obj_name)s" has been deleted successfully.</string>
    </resources>
    ```
    All the display text must be wrapped around between `<resource>` and `</resource>`. For every text to be displayed, wrapped it around a `string` tag between `<string>` and `</string>`. The `name` attribute will be used to identify the name of the text. The text between the opening and closing tag will be the text displayed to the user.

1. To display the text to the user, for example,  `INVALID_PASSWORD`, first import the module.
    ```python
    from core.lang import get_str]
    ```
    Then, call the `get_str` function.
    ```python
    get_str('INVALID_PASSWORD')
    ```
    Some messages may require additional information to be added under a different context. In these cases, a string template is supported. For example, the `ADDED` string uses a string template. The place where it is expected to be replaced should be wrapped inside `%()`. Provide a meaningful name to the placeholder. In this case, `obj_name` is used. When calling `get_str`, provide the text to be replaced in the placeholder as a keyword argument. For example
    ```python
    get_str('ADDED', obj_name=obj.name)
    ```

1. To add another, language, for example, `zh-CN`, add a `zh-CN.xml` and add the following code
    ```xml
    <resources>
        <string name="INTERNAL_ERROR">服务器内部错误。</string>
        <string name="NOT_FOUND">找不到资源。</string>
        <string name="INVALID_USERNAME">无效用户名。</string>
        <string name="INVALID_PASSWORD">密码无效。</string>
        <string name="ADDED">"%(obj_name)s"已创建。</string>
        <string name="DELETED">"%(obj_name)s"已删除</string>
    </resources>
    ```

1. Define the default language in the configuration. In this case, `en-US` will be used as the default language.
    ```json
    "DEFAULT_LANGUAGE": "en-US"
    ```
    Thus, any time a string is not found in the resource file for that language, the language system will fall back to query in the default language system. If the string is not found in the default language, an empty string will be returned.

1. To support multiple languages, the frontend can be adapted to changing the cookie `lang`. When a request is sent to the server, the cookie is sent along with the request. When the server receives the request, as long as the returned message calls the `get_str` function, the correct string for that language will be displayed. If the request contains an unknown language to the server, the default language will be used.

## Credits
- Flask-Language 0.1.0

## Author
- Bowen WU (@RapDoodle)