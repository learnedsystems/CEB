FROM postgres:12.4

RUN apt-get update && apt-get install -y \
    wget \
    tar \
    git \
    gcc \
    build-essential \
    postgresql-server-dev-12

# this will be passed in from the command line
ARG LCARD_USER
#ENV LCARD_USER=$LCARD_USER
ENV POSTGRES_USER=$LCARD_USER
ENV POSTGRES_PASSWORD="password"

# scripts in this directory will be run by the postgres docker init script

COPY ./init_db.sh /docker-entrypoint-initdb.d/_init_db.sh

USER root
ADD install_pg_hint.sh /
RUN chmod +x /install_pg_hint.sh
RUN /install_pg_hint.sh

ADD imdb_setup.sh /
ADD stack_setup.sh /
RUN chmod +x /imdb_setup.sh
RUN chmod +x /stack_setup.sh

# not running it right away because we want to restart docker container so
# postgresql conf variables take effect


