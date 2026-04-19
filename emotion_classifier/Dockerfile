FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install fastapi uvicorn httpx
RUN python -m spacy download en_core_web_sm

COPY data_pipeline.py chunker.py inference.py classifier_api.py ./

# emotion_model/ is mounted as a volume at runtime
CMD ["uvicorn", "classifier_api:app", "--host", "0.0.0.0", "--port", "8001"]
