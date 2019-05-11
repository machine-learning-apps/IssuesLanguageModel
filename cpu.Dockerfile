FROM python:3.7-slim-stretch

RUN apt-get update 
RUN apt-get upgrade -y
RUN apt-get install --reinstall build-essential -y
RUN apt install -y gcc g++

ENV CXXFLAGS="-std=c++11"
ENV CFLAGS="-std=c99"

RUN pip3 install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
RUN pip3 install torchvision

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY mdparse.py .
COPY notebooks notebooks/

EXPOSE 8823

CMD ["jupyter notebook --no-browser --allow-root --port=8823 --NotebookApp.token='$pass'"]