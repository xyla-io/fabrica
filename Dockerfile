FROM ubuntu:bionic

RUN apt-get update --fix-missing

# install PostgreSQL

RUN apt-get install -y postgresql-client postgresql-server-dev-all

# install git

RUN apt-get install -y git

# install tools

RUN apt-get install -y \
    fonts-liberation libappindicator3-1 libasound2 libatk-bridge2.0-0 \
    libnspr4 libnss3 lsb-release xdg-utils libxss1 libdbus-glib-1-2 \
    curl unzip wget \
    xvfb w3m imagemagick \
    vim

# install Python 3.7

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.7 python3.7-venv python3.7-dev python3-pip

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONUNBUFFERED=1

ENV APP_HOME /usr/src/app
WORKDIR /$APP_HOME

COPY fabrica $APP_HOME/
COPY data_layer $APP_HOME/data_layer
COPY moda $APP_HOME/moda
COPY subir $APP_HOME/subir

RUN mv $APP_HOME/config/local_docker_sql_config.py $APP_HOME/config/local_sql_config.py
RUN $APP_HOME/docker_install.sh

ENTRYPOINT [ "python3.7", "fabrica.py" ]
CMD [ "docker", "setup" ]
