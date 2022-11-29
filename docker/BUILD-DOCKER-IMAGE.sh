#!/bin/bash

IMAGE_NAME=roomba_hack
TAG_NAME=latest
BASE_IMAGE=nvidia/cudagl:11.3.0-devel-ubuntu20.04
DOCKERFILE_NAME=Dockerfile

dpkg -s nvidia-container-runtime > /dev/null 2>&1
BASE_IMAGE=ubuntu:20.04

docker build . -f ${DOCKERFILE_NAME} -t ${IMAGE_NAME}:${TAG_NAME} --build-arg BASE_IMAGE=${BASE_IMAGE}
