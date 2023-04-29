# LoRA Playground
LoRA Playground is a project that provides a self-hosted front-end to project [lora](https://github.com/cloneofsimo/lora).

## Screenshots
<img src="screenshots/01.jpg?raw=true" height="250">
<img src="screenshots/02.jpg?raw=true" height="250">
<img src="screenshots/03.jpg?raw=true" height="250">
<img src="screenshots/04.jpg?raw=true" height="250">
<img src="screenshots/05.jpg?raw=true" height="250">

## Disclaimer
Due to the time constraint, this project was built within 24 hours. The code may contain vulneraibilities that hackers can expliot. I highly recommend not to deploy the project to a publically accessible server. The project is still under development (after I finish my assignments and taking the finals). If you are interested in the project, considering star to the project ;)

## Installation guide

1. Install [Anaconda](https://www.anaconda.com/products/distribution).

1. Create a virtual environemnt (recommended) and activate the environment.

  ```bash
  conda create -n lora python==3.10 -y
  conda activate lora
  ```

1. Install PyTorch

  If you have a GPU with CUDA support

  ```bash
  conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
  ```

  If you wish to run it on a machine without NVIDIA graphic card, install the CPU only version.

  ```bash 
  conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
  ```

1. Install other dependencies with `pip`.

  ```bash 
  pip install -r requirements.txt
  ```

1. Download the pretrained models and extract according to the folder structure.

1. Populate the database (by default, run with SQLite3).

  ```bash
  python populate.py
  ```

  By default, several users are created. They are `admin`, `user1`, and `user2`. Their passwords are `12345678`.

1. Run the front-end (development server).

  ```
  python run_web.py dev
  ```

1. Run the service-end.

  ```bash
  python run_service.py --device=cpu --app-config=dev
  ```

1. Access the web with `http://127.0.0.1`. The port number can be changed in `configurations/dev.json`. Note: This is just for development purposes, do not deploy the application like this. Instead, use a WSGI server.

## Credits
Part of the code of this project is based on the open-source development framework, Mangee Flask-RESTful. The framework was developed by Bohui WU (@RapDoodle, one of our team members). The complete source code of the framework can be found in https://github.com/RapDoodle/mangee-flask-restful.

## License
The project is licensed under the MIT License.
