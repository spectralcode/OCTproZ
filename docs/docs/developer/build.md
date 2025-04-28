# How to Build OCTproZ

## Clone Repository

You can clone OCTproZ with or without plugin submodules.

### Clone with all plugins

```bash
git clone --recurse-submodules https://github.com/spectralcode/OCTproZ.git
```

### Clone without plugins

```bash
git clone https://github.com/spectralcode/OCTproZ.git
```

To initialize a specific **existing** plugin (already part of OCTproZ and tracked in `.gitmodules`):

```bash
git submodule update --init octproz_project/octproz_plugins/<plugin-folder>
```

Example for the Phase Extraction Extension:

```bash
git submodule update --init octproz_project/octproz_plugins/octproz-phase-extraction-extension
```

### Add a new plugin manually

If the plugin is **not yet listed in `.gitmodules`**, you can add it manually:

```bash
git submodule add <plugin-repo-url> octproz_project/octproz_plugins/<plugin-folder>
```

### Pull updates (main repo + plugins)

```bash
git pull --recurse-submodules
git submodule update --init --recursive
```

### (Optional) Update plugins to latest versions

If you want the **latest plugin commits** (not just the versions pinned by the main repo):

```bash
git submodule update --remote --merge
```

## Compiling

!!! info "Requirements"
    Building OCTproZ from source requires:
    
    - Installation of [Qt 5](https://www.qt.io/offline-installers) (version 5.10.1 or newer)
    - Installation of [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (version 8 or newer)
    - **Windows:** MSVC compiler that is compatible with your CUDA version (see [CUDA installation guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#system-requirements))
    - **Linux:** Development environment that is compatible with your CUDA version (see [CUDA installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements))

### Build Steps

1. Clone/Download the OCTproZ project. The destination path should not contain any spaces!
2. Start Qt Creator and open [octproz_project.pro](octproz_project/octproz_project.pro)
3. Configure project by selecting appropriate kit in Qt Creator (on Windows you need the MSVC compiler)
4. Change the CUDA architecture flags in [cuda.pri](octproz_project/octproz/pri/cuda.pri) if necessary for your hardware ([more info](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/))
5. Build _octproz_project_ (_right click on _octproz_project_  -> Build_)
6. Run OCTproZ (right click on _octproz_project_ __or__ right click on _octproz_ -> Run)

!!! note
    The project file _octproz_project.pro_ ensures that the devkit is compiled first as it generates a folder with files that are used by OCTproZ and any plug-in at compile time.

## Installing Development Tools

### Windows

!!! tip "Video Tutorial"
    [Watch the video tutorial for Windows setup](https://www.youtube.com/watch?v=DHB3NX_P1vk)

OCTproZ can be compiled with different versions of Qt, CUDA and MSVC. One of the easiest setups is possible with:

- [Visual Studio 2017](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=15)
- [CUDA 11.5.2](https://developer.nvidia.com/cuda-toolkit-archive)
- [Qt 15.12.12](https://download.qt.io/official_releases/qt/5.12/5.12.12/qt-opensource-windows-x86-5.12.12.exe)

**1. Install Visual Studio 2017**

1. Download [Visual Studio 2017 installer](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=15)
2. Start Visual Studio 2017 installer
3. Select _Desktop development with C++_. (This will install MSVC v141 and Windows 10 SDK) and start installation process
4. When installation is done, restart Windows
5. Open "Add or remove programs" in Windows system settings
6. Search for „Windows Software Development Kit"
7. Click on _Modify_ → _Change_ → _Next_ and select _Debugging Tools for Windows_. Click on _Change_. This will install the debugger that can be used later within Qt Creator. 

**2. Install CUDA**

1. Download [CUDA 11.5.2](https://developer.nvidia.com/cuda-toolkit-archive)
2. Start CUDA installer and follow instructions on screen

**3. Install Qt 5.12.12**

1. Download [Qt 15.12.12 offline installer](https://download.qt.io/official_releases/qt/5.12/5.12.12/qt-opensource-windows-x86-5.12.12.exe)
2. Disconnect from internet to skip Qt login screen
3. Start Qt installer and follow instructions on screen
4. In the _Select Components_ screen you only need _Qt 5.12.12_ → _MSVC 2017 64-bit_ and _Developer and Designer Tools_ → _Qt Creator 5.02 CDB Debugger Support_. Everything else can be unchecked.
5. Finish installation procedure and you are done!

### Linux (Debian-based Systems)

The following instructions have been tested with Ubuntu 18.04.

**1. Install Qt 5**

Open a terminal (ctrl + alt + t) and type the following commands:

```bash
sudo apt-get install build-essential
sudo apt-get install qtcreator
sudo apt-get install qt5-default
```

Qt documentation and examples (recommended but not required):

```bash
sudo apt-get install qt5-doc
sudo apt-get install qt5-doc-html qtbase5-doc-html
sudo apt-get install qtbase5-examples
```

**2. Install CUDA**

!!! warning "Important"
    Follow the [CUDA installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux) carefully.

When you get to the post-installation actions, add paths to the PATH variable by modifying the `.bashrc` file:

```bash
nano /home/$USER/.bashrc
```

Insert the CUDA relevant paths as stated in the CUDA installation guide at the end of the file.

To save the file press <kbd>Ctrl</kbd> + <kbd>O</kbd>, and to close the file press <kbd>Ctrl</kbd> + <kbd>X</kbd>.

Verify [that the CUDA toolkit can find and communicate correctly with the CUDA-capable hardware](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#verify-installation).

Finally, [install third-party libraries](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#install-libraries):

```bash
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev \
    libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
```

That's all! Now you can compile OCTproZ by opening the project files with Qt Creator.

## NVIDIA Jetson Nano with JetPack

To compile and run OCTproZ on the Jetson Nano you need Qt with enabled _OpenGL desktop_ option.

### Building Qt from Source

**1. Install build-essential package and Qt Creator**

```bash
sudo apt-get install build-essential
sudo apt-get install qtcreator
```

**2. Install build dependencies**

Enable the "Source code" option in Software and Updates > Ubuntu Software under the "Downloadable from the Internet" section. Then install the build dependencies:

```bash
sudo apt-get build-dep qt5-default
```

**3. Get the Qt source**

```bash
git clone https://code.qt.io/qt/qt5.git
cd qt5
git checkout 5.12.10
git submodule update --init --recursive
cd ~
mkdir qt5-build
cd qt5-build
```

!!! tip "Qt Version"
    You can also use the latest Qt 5 version: `git checkout 5.15`
    It works, but has not been thoroughly tested with OCTproZ on the Jetson Nano. Use at your own risk.

**4. Configure and build**

Configure Qt for _OpenGL desktop_ (required) and remove some packages with _-skip_ (optional):

```bash
../qt5/configure -qt-xcb -opengl desktop -nomake examples -nomake tests -skip qtwebengine -skip qtandroidextras -skip qtcanvas3d -skip qtcharts -skip qtconnectivity -skip qtdatavis3d -skip qtdeclarative -skip qtpurchasing -skip qtquickcontrols -skip qtquickcontrols2 -skip qtwinextras
```

!!! note "For Qt 5.15 and later"
    The `-qt-xcb` option was removed. Use `-bundled-xcb-xinput` instead.

After configuration, verify that there is a _yes_ in the line with _Desktop OpenGL_ in the _Configure summary_. Then start the build process:

```bash
make
sudo make install
```

!!! warning "Build Time"
    Be aware that `make` will take about 5 hours on the Jetson Nano.

When everything has been successfully completed, you can start Qt Creator and build OCTproZ!

## Troubleshooting

### Windows

After installing Qt 5.12.11 with the offline installer, you may get the error message:

```
NMAKE : fatal error U1077: "C:\Program": Rückgabe-Code "0x1""
```

**Solution:**

1. Close Qt Creator
2. (Re-)install __Qt Creator 4.14.2__ with the offline installer from [here](https://download.qt.io/official_releases/qtcreator/4.14/4.14.2/)
3. Delete the file _toolchains.xml_ located at:
   ```
   C:\Users\%USERNAME%\AppData\Roaming\QtProject\qtcreator\toolchains.xml
   ```
4. Reopen Qt Creator and everything should work fine

### Jetson Nano

If you encounter the following error message after the configure step:

```
ERROR: Feature 'xcb' was enabled, but the pre-condition 'features.thread && features.xkbcommon && libs.xcb' failed.
```

**Solution:**

Install the following packages:

```bash
sudo apt-get install libx11*
sudo apt-get install libxcb-*
sudo apt-get install libxcb*
sudo apt-get install libxkbcommon-dev
sudo apt-get install libxkbcommon-x11-dev
```

After installing these packages, delete the content of the qt5-build folder and run the configure step again (see [this discussion](https://github.com/microsoft/vcpkg/issues/15150)).
