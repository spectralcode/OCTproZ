 # <img style="vertical-align:middle" img src="images/octproz_icon.png" width="64"> Building OCTproZ

OCTproZ can be build on Windows and Linux. 

# Compiling
Building OCTproZ from source requires: 
- Installation of [Qt 5](https://www.qt.io/offline-installers) (version 5.10.1 or newer)
- Installation of [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (version 8 or newer)
- __Windows:__ MSVC compiler that is compatible with your CUDA version (see [CUDA installation guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#system-requirements)) To get the MSVC compiler it is the easiest to search online where/how to get it as this changes from time to time. Pay attention that you get the right version of the MSVC compiler as described in the CUDA guide. <br>
__Linux:__ Development environment that is compatible with your CUDA version (see [CUDA installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements)) and the third-party libraries mentioned in the [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#install-libraries)


How to compile:
1. Clone/Download the OCTproZ project. The destination path should not contain any spaces!
2. Start Qt Creator and open [octproz_project.pro](octproz_project/octproz_project.pro)
3. Configure project by selectig appropriate kit in Qt Creator (on Windows you need the MSVC compiler)
4. Change the CUDA architecture flags in [cuda.pri](octproz_project/octproz/pri/cuda.pri) if necessary for your hardware ([more info](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/))
5. Build _octproz_project_ (_right click on _octproz_project_  -> Build_)
6. Run OCTproZ (right click on _octproz_project_ __or__ right click on _octproz_ -> Run)

The project file _octproz_project.pro_ ensures that the devkit is compiled first as it generates a folder with files that are used by OCTproZ and any plug-in at compile time.  </br>

# Installing development tools to build OCTproZ on Windows
[-->Video tutorial<--](https://www.youtube.com/watch?v=DHB3NX_P1vk)

OCTproZ can be compiled with different versions of Qt, CUDA and MSVC. One of the easiest setups is possible with:
- [Visual Studio 2017](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=15)
- [CUDA 11.5.2](https://developer.nvidia.com/cuda-toolkit-archive)
- [Qt 15.12.12](https://download.qt.io/official_releases/qt/5.12/5.12.12/qt-opensource-windows-x86-5.12.12.exe)
### 1. Install Visual Studio 2017
1. Download [Visual Studio 2017 installer](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=15)
2. Start Visual Studio 2017 installer
3. Select _Desktop development with C++_. (This will install MSVC v141 and Windows 10 SDK) and start installation process
4. When installation is done, restart Windows
5. Open "Add or remove programs" in Windows system settings
6. Search for „Windows Software Development Kit“
7. Click on _Modify_ → _Change_ → _Next_ and select _Debugging Tools for Windows_. Click on _Change_. This will install the debugger that can be used later within Qt Creator. 

### 2 Install CUDA:
1. Download [CUDA 11.5.2](https://developer.nvidia.com/cuda-toolkit-archive)
2. Start CUDA installer and follow instructions on screen

### 3. Install Qt 5.12.12:
1. Download [Qt 15.12.12 offline installer](https://download.qt.io/official_releases/qt/5.12/5.12.12/qt-opensource-windows-x86-5.12.12.exe)
2. Disconnect from internet to skip Qt login screen
3. Start Qt installer and follow instructions on screen
4. In the _Select Components_ screen you only need _Qt 5.12.12_ → _MSVC 2017 64-bit_ and _Developer and Designer Tools_ → _Qt Creator 5.02 CDB Debugger Support_. Everything else can be unchecked.
5. Finish installation procedure and you are done!

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

After installing Qt 5.12.11 with the offline installer, you may get the error message:

```
NMAKE : fatal error U1077: "C:\Program": Rückgabe-Code "0x1""
```

One way to solve this issue is to close Qt Creator and (re-)install __Qt Creator 4.14.2__ with the offline installer form here:
https://download.qt.io/official_releases/qtcreator/4.14/4.14.2/

Then delete the file _toolchains.xml_. You can find the file here:
```
C:\Users\%USERNAME%\AppData\Roaming\QtProject\qtcreator\toolchains.xml
```

After these steps reopen Qt Creator and everything should work fine.