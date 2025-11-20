FROM python:3.12.7-slim

WORKDIR /code

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY credential.json ./credential.json
COPY ./app ./app

EXPOSE 5004

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5004"]