#!/bin/bash

sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.8 python3.8-dev -y
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10
sudo apt install python3-pip -y
python -m pip install pip

cat >> ~/.bashrc << EOT
if [ -d "$HOME/.local/bin" ] ; then
    PATH="$HOME/.local/bin:$PATH"
fi
EOT

