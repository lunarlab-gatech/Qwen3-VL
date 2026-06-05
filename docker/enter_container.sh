if [ "$(docker inspect -f '{{.State.Running}}' qwen3vl 2>/dev/null)" != "true" ]; then
    docker start qwen3vl 2>/dev/null || (echo "Container not found. Run run_container.sh first." && exit 1)
fi

docker exec -it qwen3vl /bin/bash
