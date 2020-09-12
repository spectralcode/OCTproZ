 # <img style="vertical-align:middle" img src="images/octproz_icon.png" width="64"> OCTproZ 

OCTproZ is an open source software for optical coherence tomography (OCT) processing and visualization. A plug-in system enables the integration of custom OCT systems and software modules.

<p align="center">
  <img src="images/octproz_screenshot_ubuntu.png" width="640">
</p>

The output windows in the screenshot above show OCT images of a strawberry. 

Video
--------
[Live OCT visualization: Burning an acorn with a laser](https://www.youtube.com/watch?v=d_O83kFXiRA&feature=youtu.be)


Features
--------

* **Real-time OCT processing and visualization with single GPU**  </br>
The full [OCT processing pipeline](processing.md) is implemented in [CUDA](https://developer.nvidia.com/cuda-zone) and visualization is performed with [OpenGL](https://www.opengl.org). Depending on the GPU used, OCTproZ can be used for MHz-OCT. 

* **Plug-in system** </br>
Plug-ins enable the integration of custom OCT systems and software modules. There are two kinds of plug-ins for OCTproZ: _Acquisition Systems_ and _Extensions_. An Acquisition System controls the OCT hardware and provides raw data to OCTproZ. Extensions have access to processed OCT data and can be used to extend the functionality of OCTproZ. 

* **Cross platform** </br>
OCTproZ runs on Windows and Linux. </br>
It has been successfully tested on Windows 10, Ubuntu 16.04 and Ubuntu 18.04


Processing Pipeline
--------
<p align="center">
  <img src="images/processing_pipeline_linear_v1_1_0.png" >
</p>

A detailed overview of the OCTproZ processing pipeline can be found [here](processing.md).

Performance
----------
Performance highly depends on the used computer hardware and the size of the of the OCT data. A test data set with 12 bit per sample, 1024 samples per raw A-scan, 512 A-scans per B-scan and 256 B-scans per volume was used to measure the performance on different systems:

GPU           | A-scan rate without live 3D view | A-scan rate with live 3D view
------------- | ------------- | -------------
NVIDIA Quadro K620  | ~ 300 kHz ( ~2.2 volumes/s) | ~ 250 kHz ( ~1.9 volumes/s)
NVIDIA GeForce GTX 1080 Ti  | ~ 4.8 MHz (~ 36 volumes/s) | ~ 4.0 MHz (~ 30 volumes/s)

You can find more information [here](performance.md).


Plug-ins
----------
The following plug-ins are currently available:
</br></br>
__Acquisition Systems:__
|Name | Description |
|-----|-----|
|[Virtual OCT System](octproz_project/octproz_plugins/octproz_virtual_oct_system)| Can be used to load already acquired OCT raw data from the disk|


__Extensions:__
|Name | Description |
|-----|-----|
|[Demo Extension](octproz_project/octproz_plugins/octproz_demo_extension)| This demo extension is for developers. It has no useful functionality, but the code can be used as a template for developing custom extensions.|
|[Image Statistics](https://github.com/spectralcode/ImageStatisticsExtension)| Displays useful image statistics, such as a histogram, in real time of currently acquired B-scans |

The easiest way to develop custom plug-ins is to clone/download the entire OCTproZ project, compile the DevKit and OCTproZ and use the existing examples as templates. Have a look at the [plugin developer guide](https://spectralcode.github.io/OCTproZ/developer.html). 


Download and Installation
----------
To run OCTproZ a cuda-compatible graphics card with current drivers is required.

A precompiled package for Windows (64bit) can be downloaded from:
[GitHub release section](https://github.com/spectralcode/OCTproZ/releases). Extract the zip archive and execute OCTproZ, installation is not necessary.

If you need OCTproZ for a different operating system, the easiest way is to compile it yourself. See the compiling section.

Test Dataset
----------
A test dataset that can be used with the Virtual OCT System can be downloaded from [here](https://figshare.com/articles/SSOCT_test_dataset_for_OCTproZ/12356705). 

User Manual
----------
An online version of the user manual can be found [here](https://spectralcode.github.io/OCTproZ/index.html). 

Developer Guide
----------
The plugin developer guide can be found [here](https://spectralcode.github.io/OCTproZ/developer.html). 

Compiling
---------
Compiling instructions can be found [here](BUILD.md).

Contributing
----------
Contribution guidelines can be found [here](CONTRIBUTING.md).

Long-term goals
----------
Vision and long-term goals can be found [here](vision.md).

Known issues
----------
On some Linux distributions floating dock widgets lose mouse focus when dragged. See: [Qt bug](https://bugreports.qt.io/browse/QTBUG-65640)


Publication
----------
[![status](https://joss.theoj.org/papers/539ea5d7842ff0a7607a4a405ea69730/status.svg)](https://joss.theoj.org/papers/539ea5d7842ff0a7607a4a405ea69730)

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

