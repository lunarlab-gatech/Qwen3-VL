REPO_DIR="/scratch/dbutterfield3/Research/Qwen3-VL"
DATA_DIR='/scratch/dbutterfield3/data'
XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"

# Build X11 args only when a display and auth file are available
X11_ARGS=()
if [ -n "$DISPLAY" ]; then
    X11_ARGS+=(--env="DISPLAY=$DISPLAY")
    X11_ARGS+=(--env="QT_X11_NO_MITSHM=1")
    [ -S /tmp/.X11-unix ] && X11_ARGS+=(--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw")
    if [ -f "$XAUTHORITY" ]; then
        X11_ARGS+=(--env="XAUTHORITY=/tmp/.Xauthority")
        X11_ARGS+=(--volume="$XAUTHORITY:/tmp/.Xauthority:ro")
    fi
fi

docker run --init -it \
    --name="qwen3vl" \
    --shm-size=2gb \
    --net="host" \
    --privileged \
    --gpus="all" \
    --workdir="/home/$USER/Qwen3-VL" \
    --env="NVIDIA_DISABLE_REQUIRE=1" \
    --env="XDG_RUNTIME_DIR=/tmp/runtime-$USER" \
    --env="USER_ID=$(id -u)" \
    --env="GROUP_ID=$(id -g)" \
    "${X11_ARGS[@]}" \
    --volume="$REPO_DIR:/home/$USER/Qwen3-VL" \
    --volume="$DATA_DIR:/home/$USER/data" \
    --volume="/home/$USER/.bash_aliases:/home/$USER/.bash_aliases" \
    --volume="/home/$USER/.ssh:/home/$USER/.ssh:ro" \
    --volume="/etc/localtime:/etc/localtime:ro" \
    --volume="/etc/timezone:/etc/timezone:ro" \
    --volume /tmp/runtime-$USER:/tmp/runtime-$USER \
    qwen3vl-cu128 \
    /bin/bash
