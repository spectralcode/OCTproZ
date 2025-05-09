 # <img style="vertical-align:middle" img src="images/octproz_icon.png" width="64"> OCTproZ 

OCTproZ is an open source software for optical coherence tomography (OCT) processing and visualization. A plug-in system enables the integration of custom OCT systems and software modules.

<p align="center">
  <img src="images/octproz_screenshot_ubuntu_cropped_shadow.png">
</p>

The output windows in the screenshot above show OCT images of a strawberry. 


Video
--------
[Live OCT visualization with OCTproZ: Laser cutting a spring onion](https://www.youtube.com/watch?v=3zHdpnR9zSM)


Screenshots
--------
| B-scan and volume rendering with LUT  | B-scan, en face view and 1D plot |
| ------------- | ------------- |
| <img src="images/octproz_screenshot_jetson_nano_3d_compact_cropped_shadow.png">  | <img src="images/octproz_screenshot_jetson_nano_1d_cropped_shadow.png"  >  |


Features
--------

* **Real-time OCT processing and visualization with single GPU**  </br>
The full [OCT processing pipeline](https://spectralcode.github.io/OCTproZ/site/processing.html) is implemented in [CUDA](https://developer.nvidia.com/cuda-zone) and visualization is performed with [OpenGL](https://www.opengl.org). Depending on the GPU used, OCTproZ can be used for MHz-OCT. 

* **Plug-in system** </br>
Plug-ins enable the integration of custom OCT systems and software modules. There are two kinds of plug-ins for OCTproZ: _Acquisition Systems_ and _Extensions_. An Acquisition System controls the OCT hardware and provides raw data to OCTproZ. Extensions have access to processed OCT data and can be used to extend the functionality of OCTproZ. 

* **Cross platform** </br>
OCTproZ runs on Windows and Linux. </br>
It has been successfully tested on Windows 10, Ubuntu 16.04, Ubuntu 18.04 and JetPack 4.4.1 (Jetson Nano 4 GB)


Processing Pipeline
--------
<p align="center">
  <img src="images/octproz_processing_compact.png" >
</p>

A detailed overview of the OCTproZ processing pipeline can be found [here](https://spectralcode.github.io/OCTproZ/site/processing.html).

Performance
----------
Performance highly depends on the used computer hardware and the size of the of the OCT data. A test data set with 12 bit per sample, 1024 samples per raw A-scan, 512 A-scans per B-scan and 256 B-scans per volume was used to measure the performance on different systems:

|GPU|A-scan rate without live 3D view|A-scan rate with live 3D view|
|---|---|---|
|Jetson Nano|~181 kHz (~1.4 volumes/s)|~94 kHz (~0.7 volumes/s)|
|GeForce GTX 1080|~3.40 MHz (~26 volumes/s)|~2.93 MHz (~22 volumes/s)|

You can find more information [here](performance/v180/performance_v180.md).

Download and Installation
----------
To run OCTproZ a cuda-compatible graphics card with current drivers is required.

A precompiled package for Windows (64bit) can be downloaded from:
[GitHub release section](https://github.com/spectralcode/OCTproZ/releases). Extract the zip archive and execute OCTproZ, installation is not necessary.

If you need OCTproZ for a different operating system, the easiest way is to compile it yourself. See the compiling section.

Plugins
----------
An overview of currently available plugins can be found [here](https://spectralcode.github.io/OCTproZ/site/plugins.html). 

Test Dataset
----------
A test dataset that can be used with the Virtual OCT System can be downloaded from [here](https://figshare.com/articles/SSOCT_test_dataset_for_OCTproZ/12356705). 

User Manual
----------
An online version of the user manual can be found [here](https://spectralcode.github.io/OCTproZ/site/index.html). 

Developer Guide
----------
The plugin developer guide can be found [here](https://spectralcode.github.io/OCTproZ/site/developer/index.html). 

Compiling
---------
Compiling instructions can be found [here](BUILD.md).

Contributing
----------
Contribution guidelines can be found [here](CONTRIBUTING.md).

Long-term goals
----------
Vision and long-term goals can be found [here](vision.md).



How to cite
----------
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02580/status.svg)](https://doi.org/10.21105/joss.02580)


BibTeX:
```
@article{Zabic2020,
  doi = {10.21105/joss.02580},
  url = {https://doi.org/10.21105/joss.02580},
  year = {2020},
  publisher = {The Open Journal},
  volume = {5},
  number = {54},
  pages = {2580},
  author = {Miroslav Zabic and Ben Matthias and Alexander Heisterkamp and Tammo Ripken},
  title = {Open Source Optical Coherence Tomography Software},
  journal = {Journal of Open Source Software}
}
```


Other open source OCT signal processing projects
----------
Here is a list of other projects you should check out:

| Name | Info |
|------|------|
| [ABC-OCT](https://github.com/hn-88/FDOCT) | C++ |
| [fdoct-gpu-code](https://code.google.com/archive/p/fdoct-gpu-code/) | Cuda |
| [myOCT](https://github.com/MyYo/myOCT) | Matlab |
| [OCTSharp](https://github.com/OCTSharpImaging/OCTSharp) | Cuda, C# |
| [OCTview](https://github.com/sstucker/OCTview) | C++, Python |
| [oct-cbort](https://github.com/CBORT-NCBIB/oct-cbort) | Python |
| [octlab](https://github.com/rosmir/octlab) | C/C++, LabView |
| [PyOCT](https://github.com/yuechuanlin-cw/PyOCT) | Python |
| [vortex](https://www.vortex-oct.dev/) | Cuda, C++, with Python bindings |


License
----------
OCTproZ is licensed under [GPLv3](LICENSE).</br>
The DevKit is licensed under [MIT license](octproz_project/octproz_devkit/LICENSE).
