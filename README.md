# Yolov5 running on TorchServe (GPU compatible) !

This is a docker image for Yolo V5 to be run with TorchServe (http server with high performance and using jitted version of the network).


## Usage

1) Build the torchserve image locally if using a GPU (error with the dockerhub one):
`Build the image torchserve locally for GPU before running this (cf github torchserve)`
 `https://github.com/pytorch/serve/tree/master/docker`
 
 Note: for CPU only, you can take the image from docker-hub directly, it should work fine.
 
2) After trainning a yolo v5 model on COLAB, move the "weights.pt" to the ressources folder and modify the name of your weights.pt file in the Dockerfile (line 20 and line 22)

2,bis)  Modify "index_to_name.json" to match your classes.

3) (Optional) you can modify the `batch size` in the Dockerfile (line 20) and in the `torchserve_handler.py` (line 18) 
 

4) The docker image is ready to be built and used:

`docker build . -t "your_tag:your_version"`

`docker run "your_tag:your_version"`


## Note:

For the docker-compose, you might have an issue with the GPU:
- check that you have nvidia-docker installed
- make a change in config to force GPU usage
