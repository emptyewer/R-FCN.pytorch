ARG BASEIMAGE=floydhub/pytorch:0.3.1-gpu.cuda9cudnn7-py3.35
FROM ${BASEIMAGE}

RUN apt-get update && apt-get install -y nano wget libopencv-dev python-opencv

RUN gpg --keyserver ha.pool.sks-keyservers.net --recv-keys B42F6819007F00F88E364FD4036A9C25BF357DD4
RUN curl -o /usr/local/bin/gosu -SL "https://github.com/tianon/gosu/releases/download/1.4/gosu-$(dpkg\
 --print-architecture)" \
    && curl -o /usr/local/bin/gosu.asc -SL "https://github.com/tianon/gosu/releases/download/1.4/gosu\
-$(dpkg --print-architecture).asc" \
    && gpg --verify /usr/local/bin/gosu.asc \
    && rm /usr/local/bin/gosu.asc \
    && chmod +x /usr/local/bin/gosu

ENV INSTALL_ROOT /opt

# RFCN
ENV RFCN_ROOT ${INSTALL_ROOT}/R-FCN.pytorch
ARG RFCN_URL=https://github.com/arvindmvepa/R-FCN.pytorch.git
ARG RFCN_BRANCH=kaggle1
RUN git clone --branch ${RFCN_BRANCH} ${RFCN_URL} $RFCN_ROOT
RUN mkdir -p $RFCN_ROOT/data && pip install pyyaml sympy h5py cython cffi numpy scipy menpo matplotli\
b opencv-python opencv-contrib-python easydict pandas pydicom

ENV PYTHONPATH $RFCN_ROOT/lib:$PYTHONPATH
WORKDIR $RFCN_ROOT/lib
RUN ./make.sh
RUN ./make.sh

WORKDIR /opt/R-FCN.pytorch