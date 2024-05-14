FROM python:3.11

RUN apt-get update && apt-get install -y libhdf5-dev

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./app /code/app

EXPOSE 8000

CMD [ "uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
