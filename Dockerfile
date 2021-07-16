FROM zekunli/zekun-keras-gpu

WORKDIR = /map-kurator

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt