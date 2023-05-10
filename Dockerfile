FROM tensorflow/tensorflow:latest-gpu-jupyter

COPY requirements.txt .
RUN pip install -r requirements.txt 

ENV PYTHONPATH=/code

#copy the code in at build time
#but mount over the top of it for development
COPY . /code/