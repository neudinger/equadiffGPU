wget -c https://github.com/NVIDIA/thrust/archive/refs/tags/1.16.0.tar.gz -O - | tar --strip-components 1 --directory ./include/ -xvz thrust-1.16.0/thrust &&
wget -c https://github.com/NVIDIA/cub/archive/refs/tags/1.16.0.tar.gz -O - | tar --strip-components 1 --directory ./include/ -xvz cub-1.16.0/cub