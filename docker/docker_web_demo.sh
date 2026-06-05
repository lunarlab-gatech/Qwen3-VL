#!/usr/bin/env bash
#
# This script will automatically pull docker image from DockerHub, and start a daemon container to run the Qwen-Chat web-demo.

IMAGE_NAME=qwenllm/qwenvl:qwen3vl-cu128
QWEN_CHECKPOINT_PATH='/home/dbutterfield3/Research/Qwen3-VL/models/Qwen2.5-VL-3B-Instruct-AWQ'
PORT=8901
CONTAINER_NAME=qwen3vl

function usage() {
    echo '
Usage: bash docker/docker_web_demo.sh [-i IMAGE_NAME] -c [/path/to/Qwen-Instruct] [-n CONTAINER_NAME] [--port PORT]
'
}

while [[ "$1" != "" ]]; do
    case $1 in
        -i | --image-name )
            shift
            IMAGE_NAME=$1
            ;;
        -c | --checkpoint )
            shift
            QWEN_CHECKPOINT_PATH=$1
            ;;
        -n | --container-name )
            shift
            CONTAINER_NAME=$1
            ;;
        --port )
            shift
            PORT=$1
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

if [ ! -e ${QWEN_CHECKPOINT_PATH}/config.json ]; then
    echo "Checkpoint config.json file not found in ${QWEN_CHECKPOINT_PATH}, exit."
    exit 1
fi

docker pull ${IMAGE_NAME} || {
    echo "Pulling image ${IMAGE_NAME} failed, exit."
    exit 1
}

docker run --gpus all -d --name ${CONTAINER_NAME} \
    --env="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    -v /var/run/docker.sock:/var/run/docker.sock -p ${PORT}:80 \
    --mount type=bind,source=${QWEN_CHECKPOINT_PATH},target=/data/shared/Qwen/checkpoint \
    --volume="$(cd "$(dirname "$0")/.." && pwd)/web_demo_mm.py:/vllm-workspace/web_demo_mm.py" \
    -it ${IMAGE_NAME} \
    python web_demo_mm.py --server-port 80 --server-name 0.0.0.0 -c /data/shared/Qwen/checkpoint/ --gpu-memory-utilization 0.95 --max-model-len 50000 && {
    echo "Successfully started web demo. Open 'http://localhost:${PORT}' to try!
Run \`docker logs ${CONTAINER_NAME}\` to check demo status.
Run \`docker rm -f ${CONTAINER_NAME}\` to stop and remove the demo."
}