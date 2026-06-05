# Dockerfile of qwenllm/qwenvl:qwen3vl-cu128
FROM vllm/vllm-openai:v0.11.0 as base
ENTRYPOINT []

RUN ln -sf /usr/bin/python3 /usr/bin/python && pip install --no-cache-dir \
    qwen-vl-utils==0.0.14 \
    torchcodec==0.7 \
    gradio==5.49.1 \
    transformers-stream-generator==0.0.5 \
    accelerate

# Set default user arguments which should be overwritten by docker build
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000

# Create a non-root user with matching UID/GID
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Add a custom .bashrc with warm colors and fixed hostname
RUN echo 'export PS1="\\[\\e[1;31m\\]\\u@qwen3vl_container\\[\\e[0m\\]:\\[\\e[1;33m\\]\\w\\[\\e[0m\\]\\$ "' > /home/${USERNAME}/.bashrc && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc

# Set default user & shell
USER ${USERNAME}
WORKDIR /home/${USERNAME}
SHELL ["/bin/bash", "-c"]
RUN echo "export USER=${USERNAME}" >> ~/.bashrc

# Install tmux and fonts (fonts-dejavu-core fixes Qt "no fonts" warning in cv2.imshow)
RUN sudo apt update && sudo apt install -y tmux tmuxp fonts-dejavu-core
RUN echo "set -g mouse on" >> /home/${USERNAME}/.tmux.conf