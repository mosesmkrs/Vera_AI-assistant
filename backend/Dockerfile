FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
# Inference runs on CPU in the development container. Installing the CPU wheel
# first prevents pip from resolving the large CUDA-enabled PyTorch distribution.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --no-deps torch \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY alembic.ini ./alembic.ini

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

