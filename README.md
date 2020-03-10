 # <img style="vertical-align:middle" img src="images/octproz_icon.png" width="64"> OCTproZ 

OCTproZ is an open source software for optical coherence tomography (OCT) processing and visualization. A plug-in system enables the integration of custom OCT systems and software modules.

<p align="center">
  <img src="images/octproz_screenshot_ubuntu.png" width="640">
</p>

The output windows in the screenshot above show OCT images of a strawberry. 


Features
--------

* **Real-time OCT processing and visualization with single GPU**  </br>
The full OCT processing pipeline is implemented in [CUDA](https://developer.nvidia.com/cuda-zone) and visualization is performed with [OpenGL](https://developer.nvidia.com/cuda-zone). Depending on the GPU used, OCTproZ can be used for MHz-OCT. 

* **Plug-in system** </br>
Plug-ins enable the integration of custom OCT systems and software modules. There are two kinds of plug-ins for OCTproZ: _Acquisition Systems_ and _Extensions_. An Acquisition System controls the OCT hardware and provides raw data to OCTproZ. Extensions have access to processed OCT data and can be used to extend the functionality of OCTproZ. 

* **Cross platform** </br>
OCTproZ runs on Windows and Linux. </br>
It has been successfully tested on Windows 10 and Ubuntu 16.04


New Highlights
--------

* **Live sinusoidal scan distortion correction for high speed OCT systems (since v1.1.0)**  </br>
<p align="center">
  <img src="images/sinusoidalCorrectionOnOff.png" width="260">
</p>


Performance
----------
Performance highly depends on the used computer hardware and the size of the of the OCT data. A test data set with 12 bit per sample, 1024 samples per line, 512 lines per frame and 256 frames per volume was used to measure the performance on two different systems:

GPU           | A-scan rate 
------------- | -------------
NVIDIA Quadro K620  | ~ 300 kHz ( ~2,2 volumes/s)
NVIDIA GeForce GTX 1080 Ti  | ~ 4,8 MHz (~ 36 volumes/s)


Plug-ins
----------
To develope custom plug-ins the [DevKit](octproz_devkit) needs to be used. The easiest way to develop plug-ins is to clone/download the entire OCTproZ project, compile the DevKit and OCTproZ and use the existing examples ([Virtual OCT System](octproz_virtual_oct_system), [Demo Extension](octproz_demo_extension)) as templates. </br></br>
The following plug-ins are currently available:
</br></br>
__Acquisition Systems:__
|Name | Description |
|-----|-----|
|[Virtual OCT System](octproz_virtual_oct_system)| Can be used to load already acquired OCT raw data from the disk|


__Extensions:__
|Name | Description |
|-----|-----|
|[Demo Extension](octproz_demo_extension)| This demo extension is aimed at developers. It has no useful functionality, but the code can be used as a template for developing custom extensions.|
|[Image Statistics](https://github.com/spectralcode/ImageStatisticsExtension)| Displays useful image statistics, such as a histogram, in real time of currently acquired B-scans |


Download and Installation
----------
To run OCTproZ a cuda-compatible graphics card with current drivers is required.
A precompiled package for Windows (64bit) can be downloaded from:
[GitHub release section](https://github.com/spectralcode/OCTproZ/releases).

Extract the zip archive and execute OCTproZ, installation is not necessary.

If you need OCTproZ for a different operating system, the easiest way is to compile it yourself. See the compiling section.


Compiling
---------

Compiling OCTproZ requires installation of [Qt](https://www.qt.io/) and [CUDA](https://developer.nvidia.com/cuda-zone). On Windows the MSVC2017 compiler is required. Once you have installed Qt the easiest way to compile
OCTproZ is with the QtCreator. Clone/Download the OCTproZ source files and open the .pro files with QtCreator. The DevKit needs to be compiled first as it generates a folder with files that are used by OCTproZ and any plug-ins during compile time. After successfully compiling the DevKit, OCTproZ can be compiled. </br>


Known issues
----------
- Images in the 2D views (B-scan and En Face View) are distorted when rotated. This should be easy to correct. The OpenGL part of OCTproZ should be revised anyway. Anyone who likes and has some OpenGL experience could check "glwindow2d.cpp" and give me some hints what to improve regarding OpenGL usage.
- Linux: Floating dock widgets lose mouse focus when dragged. See: [Qt bug](https://bugreports.qt.io/browse/QTBUG-65640)
- Linux: Processing can be blocked by certain GUI usage. Reason: Processing needs to run in GUI thread otherwise OpenGL output lags. This may be an OpenGL context issue. 


Contributing
----------
We strongly encourage contributions to the project. To contribute to this repository you can create [pull requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests). </br>
If you have developed a plug-in for OCTproZ and want it to be included in the precompiled package, please contact us.


Publication
----------
Coming soon. In the meantime, you can contact me at </br>
_zabic_ _</br>_
_at_</br>
_iqo_._uni_-_hannover_._de_</br>

Authors:</br>
Miroslav Zabic<sup>1, 2</sup>, Ben Matthias<sup>2</sup>, Alexander Heisterkamp<sup>1</sup>, Tammo Ripken<sup>2</sup></br>
<sup>1</sup>Institute of Quantum Optics, Leibniz University Hannover, Welfengarten 1, 30167 Hannover, Germany</br>
<sup>2</sup>Industrial and Biomedical Optics Department, Laser Zentrum Hannover e.V., Hollerithallee 8, 30419 Hannover, Germany</br>

License
----------
OCTproZ is licensed under [GPLv3](LICENSE).</br>
The DevKit is licensed under [MIT license](octproz_devkit/LICENSE).

