# map-kurator
Wrapper around Zekun's model to detect and generate annotations around map labels


## Usage
After cloning this repo and `cd ma-kurator`, run 

```
docker run -it \
-v $(pwd)/data/:/map-kurator/data \
-v $(pwd)/model:/map-kurator/model \
--rm --runtime=nvidia --gpus all  --workdir=/map-kurator \
zekunli/zekun-keras-gpu \
python model/predict_annotations.py

```

Currently, this script won't do anything; it's just a mock. I'll be adding wmts, iiif, raw file functionality later
