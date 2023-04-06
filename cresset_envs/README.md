# Using Cresset to build docker base images

We use cresset in order to build docker base images with different cuda versions and built pytorch from source using those cuda versions. Then, we upload the image to dockerhub so it can be used for the mm_bench images. In this file, we just record some of the commands used to build the images for future reference and reproducability.

## Build cresset image

1. `cp ./cresset_envs/cuda_12_0_1.env ./cresset/.env` or `cp cresset_envs/cuda_11_4_0.env ./cresset/.env`
2. `cd ./cresset`
3. `make build`

For some reason, the `make build` command will fail to run the docker image as a container using docker compose. However, this is fine since we are still able to run the images with docker run. So for now, we just care about building the images.

## Uploading the built images

1. Tag the images: `docker tag cresset:train-cuda-1140-pt2-sm80 rahulkindi/pt2-cresset:train-cuda-1140-pt2-sm80` or `docker tag cresset:train-cuda-1201-pt2-sm80 rahulkindi/pt2-cresset:train-cuda-1201-pt2-sm80`
2. Upload the image: `docker push rahulkindi/pt2-cresset:train-cuda-1140-pt2-sm80` or `docker push rahulkindi/pt2-cresset:train-cuda-1201-pt2-sm80`

## Downloading the images

They can be found at https://hub.docker.com/r/rahulkindi/pt2-cresset