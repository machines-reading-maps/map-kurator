# map-kurator
Wrapper around Zekun's model to detect map text labels


## Installation
### 1. Installing Docker
If the machine doesn't have Docker installed, you can follow instructions (for e.g., Ubuntu) here: https://docs.docker.com/engine/install/ubuntu/

In particular, here are the commands I ran to install Docker on Azure VM: 
```shell
# 1. Install prerequisites 
sudo apt-get update

sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release


# 2. Add Docker’s official GPG key:
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# 3. Set up repo
echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  
# 4. Install Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# 5. Verify that everything works
sudo docker run hello-world

# 6. Add mrm user to docker's group to allow running without sudo
usermod -a -G docker mrm
```

### 2. Download map-kurator

1. Clone this repository: 
```
git clone https://github.com/machines-reading-maps/map-kurator.git
```
2. `cd map-kurator/`

3. Build docker image, if you haven't already. 
```shell
docker build -t map-kurator .
```
This command should build the image from `Dockerfile` file in the current directory (`.`) and name the image `map-kurator`

4. **IMPORTANT** make sure the file with the model weights is available:
```shell
ls -lah data/l_weights/finetune_map_model_map_w1e50_bsize8_w1_spe200_ep50.hdf5 
#> -rwxrwxr-x 1 danf danf 183M Jul  5 18:48 data/l_weights/finetune_map_model_map_w1e50_bsize8_w1_spe200_ep50.hdf5
```
This file is over the size limit to be stored on github, hence you need to download it from [here](https://drive.google.com/file/d/1PW_wPZO54Cr5wPk44Uf8g5_gEN7UGReA/view?usp=sharing) and put it under `data/l_weights` folder.

If you are trying to run map-kurator locally and you have access to the Turing VM (and the VM is running), you can download it to your machine:
```shell
scp {USER}@{VM_HOST}:~/finetune_map_model_map_w1e50_bsize8_w1_spe200_ep50.hdf5 data/l_weights/finetune_map_model_map_w1e50_bsize8_w1_spe200_ep50.hdf5 

```

## Usage

### Input

#### WMTS

```shell
docker run -it -v $(pwd)/data/:/map-kurator/data -v $(pwd)/model:/map-kurator/model --rm  --workdir=/map-kurator map-kurator python model/predict_annotations.py wmts --url='https://wmts.maptiler.com/aHR0cDovL3dtdHMubWFwdGlsZXIuY29tL2FIUjBjSE02THk5dFlYQnpaWEpwWlhNdGRHbHNaWE5sZEhNdWN6TXVZVzFoZW05dVlYZHpMbU52YlM4eU5WOXBibU5vTDNsdmNtdHphR2x5WlM5dFpYUmhaR0YwWVM1cWMyOXUvanNvbg/wmts' --boundary='{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[-1.1248,53.9711],[-1.0592,53.9711],[-1.0592,53.9569],[-1.1248,53.9569],[-1.1248,53.9711]]]}}' --zoom=16 --dst=data/test_imgs/sample_output/ --filename=sample_filename
```

For WMTS, you can also choose to return the predicted polygons in the EPSG4326 coordinate system (lat, lng) by adding `--coord epsg4326` at the end of the above command.

#### IIIF

```shell
docker run -it -v $(pwd)/data/:/map-kurator/data -v $(pwd)/model:/map-kurator/model --rm --workdir=/map-kurator map-kurator python model/predict_annotations.py iiif --url='https://map-view.nls.uk/iiif/2/12563%2F125635459/info.json' --dst=data/test_imgs/sample_output/ --filename=sample_filename
```

#### Regular File
```shell
docker run -it -v $(pwd)/data/:/map-kurator/data -v $(pwd)/model:/map-kurator/model --rm --workdir=/map-kurator map-kurator python model/predict_annotations.py file --src={PATH_TO_INPUT_FILE} --dst=data/test_imgs/sample_output/ --filename=sample_filename
```

### Output

Assuming output directory is `--dst=$OUT_DIR` and (optional) `--filename=my_filename`, if either of the above commands ran successfully, `$OUT_DIR` will have the following files:

- `my_filename_stitched.jpg`: image that was passed to the model
  
- `my_filename_predictions.jpg`: text regions detected by the model
  
- `my_filename_annotations.json`: detected text region outlines represented as polygons (using [Web Annotation](https://www.w3.org/TR/annotation-model/) format)

If `--filename` is not provided, it will be generated automatically as a unique `uuid4()`
