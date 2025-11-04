FROM pytorch/torchserve:0.7.1-cpu
WORKDIR /home/model-server

COPY enginius_dockers/dockers/dictionary_docker/ts-requirements.txt ./
COPY enginius_dockers/dockers/dictionary_docker/ts-config.properties ./config.properties

RUN pip3 install --no-cache-dir --upgrade pip \
  && pip3 install --no-cache-dir -r ts-requirements.txt

