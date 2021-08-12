FROM zekunli/zekun-keras-gpu

WORKDIR = /map-kurator

# Install GDAL for Rasterio
RUN add-apt-repository -y ppa:ubuntugis/ppa \
 && apt-get update -y \
 && apt-get install -y python-numpy gdal-bin libgdal-dev

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt