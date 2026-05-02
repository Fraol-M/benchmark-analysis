FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_RETRIES=20

RUN apt-get update && apt-get install -y \
    build-essential cmake ninja-build git \
    libssl-dev libgmp-dev libarchive-dev \
    libpcre2-dev libedit-dev libossp-uuid-dev \
    software-properties-common \
    python3 python3-pip python3-dev \
    && add-apt-repository ppa:swi-prolog/stable \
    && apt-get update && apt-get install -y swi-prolog \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/SWI-Prolog/packages-swipy /tmp/packages-swipy && \
    cd /tmp/packages-swipy && \
    pip3 install --no-cache-dir --default-timeout=1000 --retries=20 .

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_RETRIES=20
ENV PETTA_COMMIT=e1490899cefc67c128d5311ff4861f9997674957
ENV PETTACHAINER_COMMIT=d21b93b5132a7fc8722f64d57b74fb7c3a8d1faa

RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3 python3-pip \
    libgmp10 libarchive13 libpcre2-8-0 libedit2 \
    git \
    && add-apt-repository ppa:swi-prolog/stable \
    && apt-get update && apt-get install -y swi-prolog \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --default-timeout=1000 --retries=20 --upgrade setuptools wheel
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

WORKDIR /deps
RUN git clone https://github.com/trueagi-io/PeTTa.git && \
    git -C /deps/PeTTa checkout ${PETTA_COMMIT} && \
    git clone https://github.com/rTreutlein/PeTTaChainer.git && \
    git -C /deps/PeTTaChainer checkout ${PETTACHAINER_COMMIT}

RUN cd /deps/PeTTa && \
    sed -i "/'janus-swi'/d" setup.py && \
    pip3 install --no-cache-dir --default-timeout=1000 --retries=20 .

ENV PYTHONPATH=/app/src:/deps/PeTTa/python:$PYTHONPATH

RUN cd /deps/PeTTaChainer && \
    pip3 install --no-cache-dir --default-timeout=1000 --retries=20 -e .

WORKDIR /app
COPY src/langextract_atomspace/petta_reasoner_service/requirements.txt /tmp/requirements.txt
RUN rm -rf /usr/lib/python3/dist-packages/blinker* && \
    pip3 install --no-cache-dir --default-timeout=1000 --retries=20 -r /tmp/requirements.txt

COPY src ./src

VOLUME ["/app/data"]
EXPOSE 8010

CMD ["uvicorn", "langextract_atomspace.petta_reasoner_service.app:app", "--host", "0.0.0.0", "--port", "8010"]
