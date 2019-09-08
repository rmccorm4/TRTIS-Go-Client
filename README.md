# Simple Go Client for TensorRT Inference Server via gRPC

Implemented a simple Go client from the gRPC/protobuf definitions in
<https://github.com/NVIDIA/tensorrt-inference-server/tree/master/src/core>.

## Setup

`go install github.com/rmccorm4/trtis-go-client/cmd/trtis_simple_client`

## Serving "simple" Model via Docker

```bash
# Clone repo
git clone https://github.com/NVIDIA/tensorrt-inference-server.git

# Setup "simple" model from example model_repository
cd tensorrt-inference-server/docs/examples
./fetch_models.sh

# Launch (detached) TRTIS
docker run -d -p8000:8000 -p8001:8001 -p8002:8002 \
           -it -v $(pwd)/model_repository:/models \
           nvcr.io/nvidia/tensorrtserver:19.07-py3 \
           trtserver --model-store=/models

# Check the status of the server
curl localhost:8000/api/status
```

## Using Client for Inference

```bash
# Defaults to localhost:8001
trtis_image_client -u <SERVER_IP>:8001
```

<details>
<summary>Sample Output</summary>
<p>

```bash
$ go run grpc_simple_client.go
TRTIS Health - Live: true
TRTIS Health - Ready: true
request_status:<code:SUCCESS server_id:"inference:0" request_id:3 > server_status:<id:"inference:0" version:"1.4.0" ready_state:SERVER_READY uptime_ns:39273850004 model_status:<key:"simple" value:<config:<name:"simple" platform:"tensorflow_graphdef" version_policy:<latest:<num_versions:1 > > max_batch_size:8 input:<name:"INPUT0" data_type:TYPE_INT32 dims:16 > input:<name:"INPUT1" data_type:TYPE_INT32 dims:16 > output:<name:"OUTPUT0" data_type:TYPE_INT32 dims:16 > output:<name:"OUTPUT1" data_type:TYPE_INT32 dims:16 > instance_group:<name:"simple" kind:KIND_CPU count:1 > default_model_filename:"model.graphdef" > version_status:<key:1 value:<ready_state:MODEL_READY > > > > > 

Checking Inference Outputs
--------------------------
0 + 1 = 1
0 - 1 = -1
1 + 1 = 2
1 - 1 = 0
2 + 1 = 3
2 - 1 = 1
3 + 1 = 4
3 - 1 = 2
4 + 1 = 5
4 - 1 = 3
5 + 1 = 6
5 - 1 = 4
6 + 1 = 7
6 - 1 = 5
7 + 1 = 8
7 - 1 = 6
8 + 1 = 9
8 - 1 = 7
9 + 1 = 10
9 - 1 = 8
10 + 1 = 11
10 - 1 = 9
11 + 1 = 12
11 - 1 = 10
12 + 1 = 13
12 - 1 = 11
13 + 1 = 14
13 - 1 = 12
14 + 1 = 15
14 - 1 = 13
15 + 1 = 16
15 - 1 = 14
```

</p>
</details>

## Flags 

```bash
$ trtis_simple_client -h
Usage of trtis_simple_client:
  -b int
      Batch size. Default: 1. (default 1)
  -m string
      Name of model being served. (Required) (default "simple")
  -u string
      Inference Server URL. Default: localhost:8001 (default "localhost:8001")
  -x int
      Version of model. Default: Latest Version. (default -1)
```
