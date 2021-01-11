 # <img style="vertical-align:middle" img src="images/octproz_icon.png" width="64"> Building OCTproZ

OCTproZ can be build on Windows and Linux. 

# Compiling
Building OCTproZ from source requires: 
- Installation of [Qt 5](https://www.qt.io/offline-installers) (version 5.11.1 or later)
- Installation of [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (version 8 or later)
- __Windows:__ MSVC compiler that is compatible with your CUDA version (see [CUDA installation guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#system-requirements)) To get the MSVC compiler it is the easiest to search online where/how to get it as this changes from time to time. Pay attention that you get the right version of the MSVC compiler as described in the CUDA guide. <br>
__Linux:__ Development environment that is compatible with your CUDA version (see [CUDA installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements)) and the third-party libraries mentioned in the [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#install-libraries)

A more detailed Linux guide for the installation steps above can be found in the section "Installing development tools to build OCTproZ on Linux" below. 

How to compile:
1. Clone/Download the OCTproZ project. The destination path should not contain any spaces!
2. Start Qt Creator and open [octproz_project.pro](octproz_project/octproz_project.pro)
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


# OCTproZ on the NVIDIA Jetson Nano with JetPack

To compile and run OCTproZ on the Jetson Nano you need Qt with enabled _OpenGL desktop_ option.

For this Qt can be built from source on the Jetson Nano:

### 1. Install build dependencies
Enable the "Source code" option in Software and Updates > Ubuntu Software under the "Downloadable from the Internet" section.

Open a terminal (ctrl + alt + t) an install the build dependencies like this
```
sudo apt-get build-dep qt5-default
```

### 2. Get the Qt source
```
git clone https://code.qt.io/qt/qt5.git
cd qt5
git checkout 5.12.9
```
then
```
git submodule update --init --recursive
cd ~
mkdir qt5-build
cd qt5-build
```

### 3. Configure and build
In this step you configure Qt for _OpenGL desktop_ (this is necessary!) and remove some packages with _-skip_ (this is optional. Removing those packages slightly reduces the build time)

```
../qt5/configure -qt-xcb -opengl desktop -nomake examples -nomake tests -skip qtwebengine -skip qtandroidextras -skip qtcanvas3d -skip qtcharts -skip qtconnectivity -skip qtdatavis3d -skip qtdeclarative -skip qtpurchasing -skip qtquickcontrols -skip qtquickcontrols2 -skip qtwinextras
```

After the configuration was done a _Configure summary_ will be displayed. Please verify that there is a _yes_ in the line with _Desktop OpenGL_. Now you can start the build process:

```
make
sudo make install
```
Be aware  that _make_ will take about 5 hours on the Jetson Nano. 

When everything has been successfully completed you can start Qt Creator and build OCTproZ!

References:
- [wiki.qt.io Building Qt 5 from Git](https://wiki.qt.io/Building_Qt_5_from_Git)
- [stackoverflow.com Qt-default version issue on migration from RPi4 to NVIDIA Jetson Nano](https://stackoverflow.com/questions/62190967/qt-default-version-issue-on-migration-from-rpi4-to-nvidia-jetson-nano)


# Troubleshooting

## 
GitHub user Ledinor had problems compiling OCTproZ with the default Qt Creator settings. He described  what steps were necessary for him to successfully compile OCTproZ:

>QT Version: Desktop Qt 5.12.10 MSVC2017 64bit
>Compiler C/C++: Compiler 14.0 (x86_amd64)
>CUDA: cuda_9.0.176_win10
>
>1. Use jom.exe instead of cmake! Even when I checked "Use jom instead of nmake" (settings > Build & Run), Qt Ctreator did not use it. Therefore, I use it manually. So UNCHECK this jom setting.
>-	Download jom 1.1.0
>-	In Build settings, Build steps > details, instead C:\yourpath\nmake.exe use: select the directory of the jom.exe
>-	No arguments for make (delete "clear" if this is pre-set)
>-	Uncheck “disable in subdirectories” 
>-	Do the same for the "Clean Steps"
>
>2. Qt also seems to miss the location of the rc.exe (I do not know what it does). I add the rc.exe path in "Build Environment" under PATH (for me C:\Program Files (x86)\Windows Kits\10\bin\10.0.18362.0\x64)
>
>3. After building the release file I had to add multiple QT.dll from the Qt bin folder to the release folder where the OCTproZ.exe is. For me: QT5Gui.dll, QT5Widgets.dll, QT5Core.dll, QT5PrintSupport.dll
>
>4. I had a compile error that virtualoctsystemsettingsdialog.ui could not be found, anyway it was in the correct folder. The problem solves itself when I put everything on the same hard drive as the QT installation folder. (C:/ for me)
>
> -- <cite>[Ledinor](https://github.com/spectralcode/OCTproZ/issues/12#issue-783140395)</cite>