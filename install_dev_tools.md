# Installing development tools to build OCTproZ

## Debian based systems
The following instructions have been tested with Ubuntu 18.04.

### 1. Install Qt 5:
Open a terminal (ctrl + alt + t) and type the following commands:
```
sudo apt-get install build-essential
sudo apt-get install qtcreator
sudo apt-get install qt5-default
```

Qt documentation and examples which are not not required, but recommended can be installed with these commands:
```
sudo apt-get install qt5-doc
sudo apt-get install qt5-doc-html qtbase5-doc-html
sudo apt-get install qtbase5-examples
```


### 2. Install CUDA
Follow the [CUDA installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux) carefully.

When you get to the post-installation actions, you need to know that adding paths to the PATH variable can be done with the terminal-based text editor Nano by modifying the file ".bashrc".

Open .bashrc with Nano:
```
nano /home/$USER/.bashrc
```
Now insert the cuda relevant paths as statet in the cuda installation guide at the end of the file.

To save the file press on your keyboard
```
cntrl + o
```
And to close the file press
```
cntrl + x
```

After this you should verify [that the CUDA toolkit can find and communicate correctly with the CUDA-capable hardware.](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#verify-installation)

Finally you need to [install some third-party libraries](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#install-libraries):
```
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev \
    libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
```


That is all! Now you are able to compile OCTproZ by opening the OCTproZ project files with Qt Creator. 