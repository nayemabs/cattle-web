FROM gcr.io/google-appengine/python


RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
# Create a virtualenv for dependencies. This isolates these packages from
# system-level packages.
# Use -p python3 or -p python3.7 to select python version. Default is version 2.


# Setting these environment variables are the same as running
# source /env/bin/activate.
# RUN which gunicorn
RUN /opt/python3.7/bin/python3.7 -m pip install --upgrade pip
# RUN which python
ADD . /app
# RUN which python
# pip install virtualenv
# RUN /opt/python3.7/bin/python3.7 -m venv /app/env
# ENV VIRTUAL_ENV env
# ENV PATH env/bin:$PATH
RUN /opt/python3.7/bin/python3.7 -m venv /home/vmagent/app/env
ENV VIRTUAL_ENV /home/vmagent/app/env
ENV PATH /home/vmagent/app/env/bin:$PATH
# RUN which python
# # Copy the application's requirements.txt and run pip to install all
# dependencies into the virtualenv.
# RUN pip3 uninstall mmcv
# RUN pip3 uninstall mmcv-full
# RUN which gunicorn
# RUN pip3 install --upgrade pip
# ADD . /app
# RUN ls
ADD requirements.txt /app/requirements.txt
# RUN pwd
# RUN ls
# ADD . /app

RUN /home/vmagent/app/env/bin/pip install -r requirements.txt
# RUN which gunicorn
# RUN pip3 install gunicorn==20.1.0

# RUN which gunicorn
# RUN which python
# # Add the application source code.
# # ADD . /app

# # Run a WSGI server to serve the application. gunicorn must be declared as
# # a dependency in requirements.txt.
# RUN pwd
# RUN /opt/python3.7/bin/python3.7 -m venv env
ENV VIRTUAL_ENV env
ENV PATH env/bin:$PATH
ADD . /app
RUN ls /home/vmagent/app/env/bin/
# RUN celery -A main.celery worker --loglevel=info -Q cattle --concurrency=2
# CMD gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:main --timeout 960