# Configurations

The configuration module provides an easy way to manage all the profiles used throughout development and production.

## Getting Started
1. All configurations, by default, should be placed inside the `configurations` folder. To change the location of all the configurations, change the `CONFIG_PATH` variable under `/utils/constants.py`.

1. Each configuration file is a `json` file. The filename of the configuration is used when calling `create_app` from `core.startup`. In the case of spinning up a development server using with `run.py`, type
    ```bash
    $ python run.py profile_name
    ```
    For example, to spin up a development server with the development configurations `dev.json`, type
    ```bash
    $ python run.py dev
    ```

1. Configure the json file accordingly. The following code is a template for the configuration.
    ```json
    {
        "DEBUG": true,
        "SECRET_KEY": "THIS SHOULD BE CHANGED!!!",
        "ENABLE_SIMPLE_HTTP_SERVER": false,
        "RESTFUL_PREFIX": "/api",
        "HOST": "127.0.0.1",
        "PORT": 5000,
        "LOG_PATH": "./error.log",
        "LOG_FORMAT": "%(asctime)s %(levelname)s: %(message)s",
        "SQLALCHEMY_DATABASE_URI": "sqlite:///data.db",
        "SQLALCHEMY_TRACK_MODIFICATIONS": false,
        "@JWT_AUTH_URL_RULE": "@RESTFUL_PREFIX::/auth",
        "DEFAULT_LANGUAGE": "en-US",
        "@PERMANENT_SESSION_LIFETIME": {
            "type": "timedelta",
            "args": {
                "days": 31
            }
        },
        "JWT_TOKEN_LOCATION": ["cookies"],
        "JWT_COOKIE_SECURE": false
    }
    ```
    The detailed explanation of the configuration is in the following section.

## Configuration Basics

All the configurations will be stored in the application's `config` dictionary after being parsed. Configuration entries without an `@` as the prefix will be directly added to the application's `config` dictionary. For those entries starting with `@`, special parsing will be performed before adding to the `config` dictionary.

## Special Parsing

For configuration entries starting with `@`, special parsing will be done before passing into the `config` dictionary. The rules are as follows:

- The key of the entry starts with `@` and the value contains `@()::`, the placeholder will be replaced by name of the placeholder. For example, when `"RESTFUL_PREFIX": "/api",`, `"@JWT_AUTH_URL_RULE": "@RESTFUL_PREFIX::/auth"` will be parsed into `"JWT_AUTH_URL_RULE": "/api/auth"`. Please note that the `@` symbol will be removed after being parsed. In the current stage of development, only `RESTFUL_PREFIX` is supported out of security concerns.
- The key of the entry starts with `@`, the value has a type dictionary. Inside the dictionary, the `type` parameter specifies the type (alias) of the object that will be parsed into. The `args` parameter is a dictionary containing the arguments that will be parsed into Python. For example
    ```json
    "@PERMANENT_SESSION_LIFETIME": {
        "type": "timedelta",
        "args": {
            "days": 31
        }
    }
    ```
    will be parsed into a Python object `datetime.timedelta(days=31)`. Currently, it only supports `datetime.timedelta`. Feel free to implement more as you need.

## Special Parameters

Special parameters are parameters used in this framework:

- `HOST`
    The host address of the interface to which the application will bind to.

- `PORT`
    The port that the application will bind to.

- `LOG_PATH`
    The path in which the log file will be stored. For example, `./log.log` means storing the log in a file named `log.log` under the project's root folder.

- `LOG_FORMAT`
    The format of each log entry. For `%(asctime)s %(levelname)s: %(message)s`, the output result would be
    ```
    2021-03-22 21:59:25,087 WARNING: Key "internal_error" is not implemented in en-US.xml.
    ```

- `ENABLE_HTTPS`
    A boolean value indicating whether the application should be run over HTTPS. When not specified, `false` is provided.
- `HTTPS_CERT_PATH`
    Path to the certificate of the certificate. For example, `"./security/cert.pem"`. It only needs to be specified when `ENABLE_HTTPS` is set to `true`.
- `HTTPS_PRIVATE_KEY_PATH`
    Path to the private key of the certificate. For example, `"./security/privkey.pem"`. It only needs to be specified when `ENABLE_HTTPS` is set to `true`.

- `ENABLE_SIMPLE_HTTP_SERVER`
    To run a simple HTTP server to serve the front end. It should only be used in the development environment. For production use, it is not recommended to serve the frontend in this manner for the sake of performance and security concerns. According to Flask's official documentation, it is of best practice to activate your webserver's `X-Sendfile` support.

    For more information, visit:
    https://flask.palletsprojects.com/en/1.1.x/api/#flask.send_from_directory

    WARNING:
        Using this HTTP server in production mode will bring huge
        security risks and may result in the server being exploited.

- `HTTP_SERVER_INDEX_PAGE`
    The server's index page that will be served.

- `HTTP_SERVER_INDEX_REDIRECT`
    A boolean value. When enabled, accessing `http://127.0.0.1` will be redirected to `http://127.0.0.1/index.html`
    
- `HTTP_SERVER_404_REDIRECT`
    The server's default page that is shown to the user when a page or resource is not found. If not specified, the default 404 page will be shown to the user.

- `HTTP_SERVER_REWRITE_ENGINE`
    A boolean value to indicate whether or not to enable the HTTP server's rewrite engine. This should be turned on when the HTTP server needs to serve single page application like those built on `Vuejs`.

- `HTTP_SERVER_REWRITE_TO`
    The file in which the server should serve when the HTTP server's rewrite engine in on. For example, specifying as `"index.html"` would load `index.html` even if the user accesses `http://127.0.0.1/help` and `/help` is not a folder but a route of the single page application.

## Author
- Bowen WU (@RapDoodle)