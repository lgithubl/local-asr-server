ARG CUDA_VERSION=11.8.0
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu${UBUNTU_VERSION}

ARG CUDA_VERSION

LABEL ai.lgithubl.cuda.version="${CUDA_VERSION}" \
      ai.lgithubl.ctranslate2.version="3.24.0" \
      ai.lgithubl.m40.expected_compute_capability="5.2"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ASR_HOST=0.0.0.0 \
    ASR_PORT=9000 \
    ASR_BACKEND=faster-whisper \
    ASR_MODEL_PATH=/models \
    ASR_INPUT_DIR=/inputs \
    ASR_OUTPUT_DIR=/outputs \
    ASR_DEVICE=cuda \
    ASR_COMPUTE_TYPE=float32 \
    ASR_DEFAULT_LANGUAGE=ja \
    ASR_VAD_FILTER=0 \
    ASR_PRELOAD_MODEL=1

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    ffmpeg \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libgomp1 \
    libswresample-dev \
    libswscale-dev \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip \
  && python3 -m pip install wheel setuptools 'Cython<3' \
  && python3 -m pip install --no-build-isolation -r requirements.txt

COPY asr_server ./asr_server
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
  && mkdir -p /inputs /outputs /models

EXPOSE 9000
VOLUME ["/inputs", "/outputs", "/models"]

ENTRYPOINT ["/entrypoint.sh"]
