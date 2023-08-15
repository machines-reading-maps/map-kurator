FROM zekunli/zekun-keras-gpu

WORKDIR = /map-kurator


RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt install -y wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

# Install GDAL for Rasterio
RUN add-apt-repository -y ppa:ubuntugis/ppa \
 && apt-get update -y \
 && apt-get install -y python-numpy gdal-bin libgdal-dev

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt
