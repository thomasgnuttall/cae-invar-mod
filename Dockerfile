FROM       ubuntu:18.04 as builder
MAINTAINER Thomas Nuttall "https://registry.sb.upf.edu"
LABEL authors="Thomas Nuttall"
LABEL version="18.04"
LABEL description="Complex Autoencoder for Learning Invariant Signal Representations"
RUN apt-get update
RUN apt-get install -y openssh-server nmap sudo telnet sssd
RUN mkdir /var/run/sshd
RUN echo 'root:xxxxxxxxxxxx' |chpasswd
RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config
RUN mkdir /root/.ssh
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
EXPOSE 22
CMD    ["/usr/sbin/sshd", "-D"]


FROM       python:3.7.1 as app
ADD . /
RUN apt-get update && apt-get upgrade -y && apt-get install -y && apt-get -y install apt-utils gcc libpq-dev libsndfile-dev
RUN pip install -r requirements.txt