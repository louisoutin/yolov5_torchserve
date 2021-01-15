# Yolov5 running on TorchServe (GPU compatible) !

This is a dockerfile to run TorchServe for Yolo v5 object detection model. 
(TorchServe (PyTorch library) is a flexible and easy to use tool for serving deep learning models exported from PyTorch).

You just need to pass a yolov5 weights file (.pt) in the ressources folder and it will deploy a http server, ready to serve predictions.

![alt text](request_screenshot.png)


## Setting up the docker image

1) Build the torchserve image locally if using a GPU (error with the dockerhub one):
`Build the image torchserve locally for GPU before running this (cf github torchserve)`
 `https://github.com/pytorch/serve/tree/master/docker`
 
 Note: for CPU only, you can take the image from docker-hub directly, it should work fine.
 
2) After trainning a yolo v5 model on COLAB, move the `weights.pt` to the ressources folder and modify the name of your `weights.pt` file in the Dockerfile (line 20 and line 22)

3)  Modify "index_to_name.json" to match your classes.

4) (Optional) you can modify the `batch size` in the Dockerfile (line 20) and in the `torchserve_handler.py` (line 18) 
 

5) The docker image is ready to be built and used:

`docker build . -t "your_tag:your_version"`

`docker run "your_tag:your_version"`

## Getting predictions

Once the dockerimage is running, you can send POST requests to: `localhost:8080/predictions/my_model` (with `my_model` being the name of your model).

The handler in this project expect the inputs images to be sent via a Multipart form with a "key/value" form having in the keys the strings "img"+`[index]` and in the values, the bytes of each images.

Example:
-------
For a batch_size of 5, we would have the following in our Multipart form request:

```
"img1": [bytes_of_the_1st_image],
"img2": [bytes_of_the_2st_image],
"img3": [bytes_of_the_3st_image],
"img4": [bytes_of_the_4st_image],
"img5": [bytes_of_the_5st_image],
```

The returned json of the request contain a single list. Each i-th element of this list represent the i-th image detection results (represented by:
`(x1, y1, x2, y2, conf, cls)`)

There is a request example on the image of this Readme.
Note that if there is less input images than the batch size, the rest of the inference batch will be padded with zeros inputs.

## Note:

The yolov5 folder in ressources is just here to export the model to a torchscript version.
(It could be optimized to keep only the `export.py` file)

For the docker-compose, you might have an issue with the GPU:
- check that you have nvidia-docker installed
- make a change in docker-compose configs to force GPU usage (there is an issue on docker-compose github open)

If you want to run with a CPU, change the line 'cuda:0' to 'cpu' in the export.py file of yolov5

TO DO:
- For now I only tested it with GPU as this is my usecase, but later I'll try to automate the build so that it's easier to switch to CPU
- The whole repo of yolov5 is in the ressource folder, but only the export is used, I will refactor to keep only the export part (a bit tricky with dependencies)
