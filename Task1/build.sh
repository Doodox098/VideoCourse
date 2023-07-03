set -o xtrace

setup_root() {
    apt-get install -qq -y \
        python3-pip \
        python3-tk \
        cmake \
        python3-opencv

    pip3 install -qq \
        pandas \
        pytest \
        scikit-image \
        scikit-learn \
        tqdm \
        matplotlib \
        opencv-python \
        pybind11
}

setup_checker() {
    python3 -c 'import matplotlib.pyplot'
}

"$@"