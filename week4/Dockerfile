FROM svizor/zoomcamp-model:mlops-3.10.0-slim

WORKDIR /app
RUN ["mkdir", "data"]

RUN ["pip", "install", "pipenv"] 
COPY [ "Pipfile", "Pipfile.lock", "./" ]
RUN pipenv install --system --deploy

COPY notebooks/starter.py .
ENTRYPOINT ["python", "starter.py"]