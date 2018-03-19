FROM nvidia/opencl:runtime-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake libclfft-dev libtiff-dev nvidia-opencl-icd-384
COPY . /tmp/code
WORKDIR /tmp/code
RUN cmake . && make && make install
RUN cp -r /tmp/code/sample /root/sample && rm -rf /tmp/code
WORKDIR /root

CMD ["stemcl"]