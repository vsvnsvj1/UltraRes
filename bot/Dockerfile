FROM python:3.12.4-slim-bookworm

WORKDIR /app

COPY requirements/bot_requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt  \
    && rm -rf /var/lib/apt/lists/*

COPY bot/ /app/bot/

CMD ["python", "-m", "bot.__main__"]
