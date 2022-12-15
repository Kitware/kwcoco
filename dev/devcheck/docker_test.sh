#DOCKER_IMAGE=circleci/python:3.9-rc
DOCKER_IMAGE=python:3.7
docker pull $DOCKER_IMAGE

docker run -v $PWD:/io \
    --rm -it $DOCKER_IMAGE bash -c "pip install kwcoco"

docker run -v $PWD:/io \
    --rm -it $DOCKER_IMAGE bash
