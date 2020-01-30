# docker run command
# docker run -Pit -u root:root --name dlhk --runtime=nvidia -v /home/hyunkim:/home/hyunkim -e "0000" -p 8888:8888 -p 6006:6006 tensorflow/tensorflow:latest-gpu-py3

apt-get update

# OpenCV dependency       
apt-get install -y libsm6 libxext6 libxrender-dev

pip install opencv-python

pip install keras 

apt install -y git
apt-get install -y vim