FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install httpx

COPY app.py pipeline_api.py ./
COPY speaker.wav .

RUN mkdir -p output EmotionAudio

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
