#! /bin/sh

sudo apt-get update
sudo apt-get upgrade
cd ~/
if [[ -z "${CUDA_INSTALLED}" ]]; then
lspci | grep -i nvidia
uname -m && cat /etc/*release
gcc --version
sudo apt install gcc
uname -r		
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
sudo apt-get install linux-headers-$(uname -r)
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list
sudo apt-get update
sudo apt-get install cuda
sudo apt-get install nvidia-gds
cd /etc
sudo sed -i -e '$a\CUDA_INSTALLED=done' environment
echo "Cuda Install Block Completed. Rerun this script after reboot to continue setup. Rebooting in 10..."
sleep 10
sudo reboot
fi

if [[ -z "${DRIVER_INSTALLED}" ]]; then
echo "Continuing setup from cuda install"
export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
cd /lib/udev
sudo apt install vim
cd rules.d
sudo cp /lib/udev/rules.d/40-vm-hotadd.rules /etc/udev/rules.d
sudo sed -i '/SUBSYSTEM=="memory", ACTION=="add"/d' /etc/udev/rules.d/40-vm-hotadd.rules
sudo apt install git
cd ~/
sudo git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples/Samples/0_Introduction
sudo make
cat /proc/driver/nvidia/version
cd ~/
cuda-samples/Samples/1_Utilities/deviceQuery/deviceQuery
cuda-samples/Samples/1_Utilities/bandwidthTest/bandwidthTest 
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa-dev libfreeimage-dev libglfw3-dev
cd ~/Downloads
sudo snap install curl
curl -O "https://us.download.nvidia.com/tesla/535.54.03/nvidia-driver-local-repo-ubuntu2204-535.54.03_1.0-1_amd64.deb"
sudo apt-get install gdebi 
sudo cp /var/nvidia-driver-local-repo-ubuntu2204-535.54.03/nvidia-driver-local-D69FCCEA-keyring.gpg /usr/share/keyrings/
sudo gdebi nvidia-driver-local-repo-ubuntu2204-535.54.03_1.0-1_amd64.deb
cd /etc
sudo sed -i -e '$a\DRIVER_INSTALLED=done' environment
echo "Driver Install Block Completed. Rerun this script after reboot to continue setup. Rebooting in 10..."
sleep 10
sudo reboot
fi

if [[ -z "${CUDA_DONE}" ]]; then
cd ~/Downloads
sudo apt-get install zlib1g
FILE=$HOME/Downloads/cudnn-local-repo-cross-sbsa-ubuntu2204-8.9.3.28_1.0-1_all.deb
echo "YOU MUST DOWNLOAD CUDNN UBUNTU2204 CROSS SBSA BRANCH 12.x TO DOWNLOADS BEFORE CONTINUING!"
sleep 3
xdg-open "https://developer.nvidia.com/cudnn"

while [ ! -f "$FILE" ]
do
clear
echo "Waiting for download completion."
sleep 1
clear
echo "Waiting for download completion.."
sleep 1
clear 
echo "Waiting for download completion..."
sleep 1
done
echo "Resuming..."
cd ~/Downloads
sudo gdebi cudnn-local-repo-cross-sbsa-ubuntu2204-8.9.3.28_1.0-1_all.deb 
sudo cp /var/cudnn-local-repo-cross-sbsa-ubuntu2204-8.9.3.28/cudnn-local-046F9092-keyring.gpg /usr/share/keyrings/ 
cd ~/
sudo apt-get install libcudnn8=8.9.3.28-1+cuda12.1
sudo apt-get install libcudnn8-dev=8.9.3.28-1+cuda12.1
sudo apt-get install libcudnn8-samples=8.9.3.28-1+cuda12.1
cp -r /usr/src/cudnn_samples_v8/ $HOME
cd $HOME/cudnn_samples_v8/mnistCUDNN
make clean && make
./mnistCUDNN
cd /etc
sudo sed -i -e '$a\CUDA_DONE=done' environment
echo "CUDNN Install Block Completed. Rerun this script after reboot to continue setup. Rebooting in 10..."
sleep 10
sudo reboot
fi

if [[ -z "${OPENCV_COMPILED}" ]]; then
cd ~/
sudo git clone -b 4.8.0 --single-branch https://github.com/opencv/opencv
sudo git clone -b 4.8.0 --single-branch https://github.com/opencv/opencv_contrib
cd opencv
sudo mkdir build
cd build
sudo apt-get install cmake
sudo cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local  -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules/ -DBUILD_EXAMPLES=ON  -DWITH_CUDA=ON  -DBUILD_DOCS=ON -DCUDA_ARCH_BIN=6.1 -DCMAKE_CXX_FLAGS=-std=c++11 ..
sudo sed -i '114s/^[\t ]*if (weight != 1.0)/\t if ((double)weight != 1.0)/' ~/opencv/modules/dnn/src/cuda4dnn/primitives/normalize_bbox.hpp
sudo sed -i '124s/^[\t ]*if (nms_iou_threshold > 0)/\t if ((double)nms_iou_threshold > 0)/' ~/opencv/modules/dnn/src/cuda4dnn/primitives/region.hpp 
make -j$(nproc)
sudo make install
make clean
cd /etc
sudo sed -i -e '$a\OPENCV_COMPILED=done' environment
echo "Final Install Block Completed. Rebooting in 10..."
sleep 10
sudo reboot
else
sudo sed -i -e '$a\OpenCV_DIR=~/opencv/build' environment
unset OPENCV_COMPILED
unset CUDA_DONE
unset DRIVER_INSTALLED
unset CUDA_INSTALLED
sudo apt-get autoremove
thisscript=$0
shred -u ${thisscript}
fi
