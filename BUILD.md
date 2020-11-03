 # <img style="vertical-align:middle" img src="images/octproz_icon.png" width="64"> Building OCTproZ

OCTproZ can be build on Windows and Linux. 

# Compiling
Building OCTproZ from source requires: 
- Installation of [Qt 5](https://www.qt.io/offline-installers) (verion 5.11.1 or later)
- Installation of [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (version 8 or later)
- __Windows:__ MSVC compiler that is compatible with your CUDA version (see [CUDA installation guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#system-requirements)) To get the MSVC compiler it is the easiest to search online where/how to get it as this changes from time to time. Pay attention that you get the right version of the MSVC compiler as described in the CUDA guide. <br>
__Linux:__ Development environment that is compatible with your CUDA version (see [CUDA installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements)) and the third-party libraries mentioned in the [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#install-libraries)

A more detailed Linux guide for the installation steps above can be found in the section "Installing development tools to build OCTproZ on Linux" below. 

How to compile:
1. Clone/Download the OCTproZ project. The destination path should not contain any spaces!
2. Start Qt Creator and open [octproz_project.pro](octproz_project.pro)
3. Configure project by selectig appropriate kit in Qt Creator (on Windows you need the MSVC compiler)
3. Build _octproz_project_ (_right click on _octproz_project_  -> Build_)
4. Run OCTproZ (right click on _octproz_project_ __or__ right click on _octproz_ -> Run)

The project file _octproz_project.pro_ ensures that the devkit is compiled first as it generates a folder with files that are used by OCTproZ and any plug-in at compile time.  </br>


# Installing development tools to build OCTproZ on Linux

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
ctrl + o
```
And to close the file press
```
ctrl + x
```

After this you should verify [that the CUDA toolkit can find and communicate correctly with the CUDA-capable hardware.](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#verify-installation)

Finally you need to [install some third-party libraries](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#install-libraries):
```
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev \
    libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
```


That is all! Now you are able to compile OCTproZ by opening the OCTproZ project files with Qt Creator. 
