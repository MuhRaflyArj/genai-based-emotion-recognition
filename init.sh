#!/bin/bash
set -e
IMAGE_NAME="journal-api"
CONTAINER_NAME="journal-api-container"

if [ $(docker ps -q -f name=^/${CONTAINER_NAME}$) ]; then
    docker stop ${CONTAINER_NAME}
    docker rm ${CONTAINER_NAME}
fi

docker build -t ${IMAGE_NAME} .
docker run -d -p 5001:5001 --env-file ./.env --name ${CONTAINER_NAME} ${IMAGE_NAME}