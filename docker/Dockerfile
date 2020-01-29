## Before building this Dockerfile 
## make sure nvidia-docker is installed on the host envirment 
## how to install nvidia-docker
# # 레파지토리 추가
# curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
#   sudo apt-key add -
# curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/nvidia-docker.list | \
#   sudo tee /etc/apt/sources.list.d/nvidia-docker.list
# sudo apt-get update

# # 패키지 설치
# sudo apt-get install -y nvidia-docker2

# # docker 데몬 내림
# sudo pkill -SIGHUP dockerd

## /etc/docker/daemon.json  에 아래 설정을 추가
# {
#     "runtimes": {
#         "nvidia": {
#             "path": "/usr/bin/nvidia-container-runtime",
#             "runtimeArgs": []
#         }
#     }
# }

FROM tensorflow:latest-gpu-py3
MAINTAINER n31fromsouth "n31fromsouth@gmail.com"

# Create user
ARG USER_ID=n31
ENV USER_ID $USER_ID
ARG USER_PASS=pass
ENV USER_PASS $USER_PASS
ARG USER_UID=1999
ENV USER_UID $USER_UID
ARG USER_GID=1999
ENV USER_GID $USER_GID

RUN \
groupadd --system -r $USER_ID -g $USER_GID && \
adduser --system --uid=$USER_UID --gid=$USER_GID --home /home/$USER_ID --shell /bin/bash $USER_ID && \
echo $USER_ID:$USER_PASS | chpasswd && \
cp /etc/skel/.bashrc /home/$USER_ID/.bashrc && . /home/$USER_ID/.bashrc && \
cp /etc/skel/.profile /home/$USER_ID/.profile && . /home/$USER_ID/.profile && \
chown $USER_ID:$USER_ID /home/$USER_ID/.*  && \
adduser $USER_ID sudo

# login profile
COPY .bash_profile /home/$USER_ID/
RUN chown $USER_ID:$USER_ID /home/$USER_ID/.*

USER root
RUN \
echo "export LANG='en_US.UTF-8'" | tee -a  /home/$USER_ID/.bashrc  && \
echo "export LANGUAGE='en_US.UTF-8'" | tee -a  /home/$USER_ID/.bashrc  && \
echo "export LC_ALL='en_US.UTF-8'"  | tee -a   /home/$USER_ID/.bashrc  && \
echo "export TZ='Asia/Seoul'"  | tee -a   /home/$USER_ID/.bashrc  && \
echo "export TERM='xterm'"  | tee -a   /home/$USER_ID/.bashrc  && \
echo "set input-meta on" | tee -a   /home/$USER_ID/.inputrc && \
echo "set output-meta on" | tee -a  /home/$USER_ID/.inputrc && \
echo "set convert-meta off" | tee -a  /home/$USER_ID/.inputrc && \
echo "export LS_COLORS='rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arj=01;31:*.taz=01;31:*.lzh=01;31:*.lzma=01;31:*.tlz=01;31:*.txz=01;31:*.zip=01;31:*.z=01;31:*.Z=01;31:*.dz=01;31:*.gz=01;31:*.lz=01;31:*.xz=01;31:*.bz2=01;31:*.bz=01;31:*.tbz=01;31:*.tbz2=01;31:*.tz=01;31:*.deb=01;31:*.rpm=01;31:*.jar=01;31:*.war=01;31:*.ear=01;31:*.sar=01;31:*.rar=01;31:*.ace=01;31:*.zoo=01;31:*.cpio=01;31:*.7z=01;31:*.rz=01;31:*.jpg=01;35:*.jpeg=01;35:*.gif=01;35:*.bmp=01;35:*.pbm=01;35:*.pgm=01;35:*.ppm=01;35:*.tga=01;35:*.xbm=01;35:*.xpm=01;35:*.tif=01;35:*.tiff=01;35:*.png=01;35:*.svg=01;35:*.svgz=01;35:*.mng=01;35:*.pcx=01;35:*.mov=01;35:*.mpg=01;35:*.mpeg=01;35:*.m2v=01;35:*.mkv=01;35:*.webm=01;35:*.ogm=01;35:*.mp4=01;35:*.m4v=01;35:*.mp4v=01;35:*.vob=01;35:*.qt=01;35:*.nuv=01;35:*.wmv=01;35:*.asf=01;35:*.rm=01;35:*.rmvb=01;35:*.flc=01;35:*.avi=01;35:*.fli=01;35:*.flv=01;35:*.gl=01;35:*.dl=01;35:*.xcf=01;35:*.xwd=01;35:*.yuv=01;35:*.cgm=01;35:*.emf=01;35:*.axv=01;35:*.anx=01;35:*.ogv=01;35:*.ogx=01;35:*.aac=00;36:*.au=00;36:*.flac=00;36:*.mid=00;36:*.midi=00;36:*.mka=00;36:*.mp3=00;36:*.mpc=00;36:*.ogg=00;36:*.ra=00;36:*.wav=00;36:*.axa=00;36:*.oga=00;36:*.spx=00;36:*.xspf=00;36:'" | tee -a  /home/$USER_ID/.bashrc

################################################################################
# Python setting
################################################################################
USER root
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
RUN apt-get install -y python3-pip
RUN apt-get install -y sudo
RUN apt-get install -y protobuf-compiler python-pil python-lxml
# for OpenCV
RUN apt-get install -y libsm6 libxext6 libxrender-dev
# vim, git
RUN apt-get install -y vim 
RUN apt-get install -y git

# Upgrade pip
RUN pip3 install -U pip

# Cython
RUN pip install cython
# Contextlib2
RUN pip install contextlib2
# Jupyter
RUN pip install jupyter
# Matplotlib
RUN pip install matplotlib 
# Pandas: 
RUN pip install pandas
# tensorflow
RUN pip install tensorflow-gpu==2.0
# tqdm 
RUN pip install tqdm
# Opencv 
RUN pip install opencv-python
# Pillow
RUN pip install pillow
#sklearn
RUN pip install scikit-learn
#requests 
RUN pip install requests


# Tensorflow repository through git
WORKDIR /home/$USER_ID
################################################################################
# Jupyter setting
################################################################################

EXPOSE 8888
USER $USER_ID 
RUN jupyter notebook --generate-config

USER root
RUN \
chown -R $USER_ID:$USER_ID /home/$USER_ID/.jupyter && \
chmod -R 755 /home/$USER_ID/.jupyter
RUN \
echo "c.NotebookApp.ip = '0.0.0.0'" | tee -a /home/$USER_ID/.jupyter/jupyter_notebook_config.py && \
echo "c.NotebookApp.notebook_dir = u\"/home/$USER_ID\"" | tee -a /home/$USER_ID/.jupyter/jupyter_notebook_config.py && \
echo "c.NotebookApp.token = u\"\"" | tee -a /home/$USER_ID/.jupyter/jupyter_notebook_config.py && \
echo "c.NotebookApp.password = u\"\"" | tee -a /home/$USER_ID/.jupyter/jupyter_notebook_config.py && \
echo "c.NotebookApp.iopub_data_rate_limit = 10000000" | tee -a /home/$USER_ID/.jupyter/jupyter_notebook_config.py \
echo "c.NotebookApp.open_browser = False" | tee -a /home/$USER_ID/.jupyter/jupyter_notebook_config.py

USER $USER_ID

WORKDIR /home/$USER_ID

ENTRYPOINT [ "/bin/bash" ]