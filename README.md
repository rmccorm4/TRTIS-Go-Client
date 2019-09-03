# (Unofficial) Go Image Client for TensorRT Inference Server

## Setup

`go install github.com/rmccorm4/trtis-go-client/cmd/trtis_image_client`

## Flags 

```bash
> trtis_image_client -h

Usage of trtis_image_client:
  -a    
        Use asynchronous inference API
  -b int
        Batch size. Default is 1. (default 1)
  -c int
        Number of class predictions to report. Default is 1. (default 1)
  -i string
        Input image/directory. (Required)
  -m string
        Name of model being served. (Required)
  -s string
        Type of scaling to apply to image pixels. Default is NONE (default "NONE")
  -streaming
        Use streaming inference API
  -u string
        Inference Server URL. Default is localhost:8001 (default "localhost:8001")
  -v    
        Enable verbose output.
  -x int
        Version of model. Default is 1. (default 1)
```

## Usage

This assumes that you've separately launched TRTIS, serving the model 
`resnet50_netdef` from the [example model repository](https://github.com/NVIDIA/tensorrt-inference-server/tree/master/docs/examples/model_repository/resnet50_netdef)

Serving this model would look something like this using Docker:

```bash
# Assumes you have $(pwd)/models/resnet50_netdef/1/...
docker run -d -it --net=host -v $(pwd)/models:/models nvcr.io/nvidia/tensorrtserver:19.07-py3 trtserver --model-store=/models
```

Once the model is being served, you can use ths client like so:

```bash
trtis_image_client -m resnet50_netdef -i mug.jpg
```
