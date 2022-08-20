FROM python:3.7.13-slim-buster
MAINTAINER Arshavir Ter-Gabrielyan "tergabrielyan@gmail.com"
WORKDIR /work
COPY . /work

# Install missing packages
RUN apt-get update
RUN apt-get install -y curl unzip
RUN mkdir /downloads
RUN mkdir /provers

# Install Z3 [4.8.10]
RUN cd /downloads && curl -L -O https://github.com/Z3Prover/z3/releases/download/z3-4.8.10/z3-4.8.10-x64-ubuntu-18.04.zip
RUN cd /downloads && unzip z3-4.8.10-x64-ubuntu-18.04.zip
RUN mv /downloads/z3-4.8.10-x64-ubuntu-18.04 /provers/z3
ENV PATH="${PATH}:/provers/z3/bin"

# Install CVC4 [1.6]
RUN cd /downloads && curl -L -O http://cvc4.cs.stanford.edu/downloads/builds/x86_64-linux-opt/cvc4-1.6-x86_64-linux-opt
RUN mv /downloads/cvc4-1.6-x86_64-linux-opt /provers/cvc4
RUN chmod +x /provers/cvc4

# Install Vampire [4.4]
RUN cd /downloads && curl -L -O https://vprover.github.io/bin/vampire_z3_rel_static_release_v4.4
RUN mv /downloads/vampire_z3_rel_static_release_v4.4 /provers/vampire
RUN chmod +x /provers/vampire

# Upldate the path
ENV PATH="${PATH}:/provers"

# Install PySMT
RUN cd /downloads && curl -L https://github.com/alebugariu/pysmt/archive/refs/tags/v0.8.0-ale.tar.gz | tar xvz -C /
RUN rm -fr /work/pysmt && ln -s /pysmt-0.8.0-ale/pysmt /work/pysmt

# Install other dependencies
RUN pip3 install -e .

# Clean-up
RUN rm -fr /downloads