FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1

COPY main.py taxonomy.yaml ./

ENTRYPOINT ["python", "main.py"]
