FROM python:3.12.4-slim-bookworm

WORKDIR /app

COPY requirements/worker_requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt \
    && rm -rf /var/lib/apt/lists/*

COPY model/ /app/model/
COPY worker/ /app/worker/

CMD ["python","-m", "worker.__main__"]
