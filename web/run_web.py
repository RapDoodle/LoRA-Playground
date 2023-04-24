# -*- coding: utf-8 -*-
import sys

# Core modules
from core.startup import run
from core.startup import create_app


if __name__ == '__main__':
    # Get the app according to the provided profile
    app = create_app(name=__name__,
        config_name=sys.argv[1] if len(sys.argv) > 1 else 'default')
    
    # Spin up the development server
    run(app)
