FROM mcr.microsoft.com/devcontainers/python:3.11 

WORKDIR /app
# COPY . /app

RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir evo
RUN pip install --no-cache-dir scipy
# CMD ["python", "preintegrator.py"]

