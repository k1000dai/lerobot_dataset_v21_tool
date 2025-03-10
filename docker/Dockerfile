FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Configure image
ARG PYTHON_VERSION=3.10
ARG DEBIAN_FRONTEND=noninteractive

# Install build tools and compilers
RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    build-essential \
    cmake \
    libtool \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gawk \
    git \
    git-lfs \
    grep \
    htop \
    less \
    nano \
    screen \
    sed \
    texinfo \
    tmux \
    tree \
    unzip \
    util-linux \
    vim \
    wget \
    zip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install graphics libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1-mesa \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /root/hsr_data_converter

# Setup Python: Create a symbolic link for easier access
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install uv (Fast Python package manager)
ENV PATH="/root/.local/bin:$PATH"
RUN wget -qO- https://astral.sh/uv/install.sh | sh

# Install GitHub CLI tool (gh)
RUN (type -p wget >/dev/null || (apt-get update && apt-get install wget -y)) \
    && mkdir -p -m 755 /etc/apt/keyrings \
    && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install gh -y \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install FFmpeg
COPY ./scripts/ffmpeg_installer.sh ./ffmpeg_installer.sh
RUN chmod +x ./ffmpeg_installer.sh && ./ffmpeg_installer.sh

# Install packages for rerun
RUN apt-get update && apt-get install -y --no-install-recommends \
    libusb-1.0-0-dev \
    portaudio19-dev \
    qtbase5-dev \
    qtbase5-dev-tools \
    libx11-dev \
    libxcursor-dev \
    libx11-xcb-dev \
    libxi-dev \
    libxkbcommon-x11-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Clean up and remove unnecessary files
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*