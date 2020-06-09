---
title: 'Open Source Optical Coherence Tomography Software'
tags:
  - C++
  - CUDA
  - Optical Coherence Tomography
authors:
  - name: Miroslav Zabic
    orcid: 0000-0002-4494-6127
    affiliation: "1, 2"
  - name: Ben Matthias
    affiliation: 2 
  - name: Alexander Heisterkamp
    affiliation: 1
  - name: Tammo Ripken
    affiliation: 2        
affiliations:
 - name: Institute of Quantum Optics, Leibniz University Hannover, Welfengarten 1, 30167 Hannover, Germany
   index: 1
 - name: Industrial and Biomedical Optics Department, Laser Zentrum Hannover e.V., Hollerithallee 8, 30419 Hannover, Germany
   index: 2
date: 22 May 2020
bibliography: paper.bib
---

# Summary

Optical coherence tomography (OCT) is a non-invasive imaging technique that is often described as the optical equivalent to ultrasound imaging. 
The basic building block of OCT acquisitions is an optical interference pattern that can be processed into a depth profile, which is also called A-scan. Several adjacent A-scans can be merged into a cross-sectional image. Most research that incorporates OCT requires a software solution for processing of the acquired raw data.

Here we present an open source software for OCT processing with an easy to use graphical user interface. The implemented OCT processing pipeline enables A-scan processing rates in the MHz range. Custom OCT systems, or any other source of Fourier Domain OCT raw data, can be integrated via a developed plug-in system, which also allows the development of custom post processing modules.

# 1. Introduction

Optical coherence tomography (OCT) is a non-invasive imaging technique used primarily in the medical field, especially in ophthalmology. The core element of any OCT system is an optical interferometer that generates a spectral fringe pattern by combining reference beam and the backscattered light from a sample. To obtain an interpretable image from this acquired raw OCT signal several processing steps are necessary, whereby the inverse Fourier transform represents an essential step. As the possible acquisition speed for raw OCT data has increased constantly, more sophisticated methods were needed for processing and live visualization of the acquired OCT data. A particularly impressive setup was presented by Choi et al. [@choi2012spectral] that utilizes twenty FPGA-modules for real-time OCT signal processing and a graphics processing unit (GPU) for volume rendering. Nowadays, processing is typically done on graphics cards [@zhang2010real; @rasakanthan2011processing; @sylwestrzak2012four; @jian2013graphics; @wieser2014high], not FPGAs, because implementing algorithms on GPUs is more flexible and takes less time. [@li2011scalable] Most of the publication that describe OCT GPU processing do not provide the actual software implementation.
A commendable exemption is the GPU accelerated OCT processing pipeline published by Jian et al. The associated source code, which demonstrates an implementation of OCT data processing and visualization and does not include any advanced features such a graphical user interface (GUI), already consists of several thousand lines. Thus, the most time consuming task of Fourier Domain OCT (FD-OCT) system development is not the optical setup, but the software development. The software can be separated into hardware control and signal processing, whereby the former being a highly individual, hardware-dependent software module and the latter being a generic software module, which is almost identical for many systems. To drastically reduce OCT system development time, we present OCTproZ, an open source OCT processing software that can easily be extended, via a plug-in system, for many different hardware setups. In this paper we give a brief overview of the key functionality and structure of the software.

# 2. Basic overview of OCTproZ

OCTproZ performs live signal processing and visualization of OCT data. It is written in C++, uses the cross-platform application framework Qt for the GUI and utilizes Nvidia’s computer unified device architecture (CUDA) for GPU parallel computing. A screenshot of the application can be seen in Fig. \ref{fig:screenshot}.

 ![Screenshot of OCTproZ v1.0. Processing settings visible in the left panel can be changed before processing is started or while processing is in progress. Processed data is live visualized in 2D as cross sectional images (B-scan and en face view) and in 3D as interactive volume rendering. The live view shows a piece of wood with a couple layers of tape and a laser burned hole. \label{fig:screenshot}](figures/20191122_screenshot3d.png)


The software can be separated into three parts: main application, development kit (DevKit) and plug-ins. The main application, OCTproZ itself, contains the logic for GUI, processing and visualization. The DevKit, which is implemented as static library, provides the necessary interface for plugin development. Plug-ins can be one of two kinds: “Acquisition Systems” or “Extensions”. The former represent software implementations of physical or virtual OCT system hardware, the later are software modules that extend the functionality of an OCT system (e.g. software control of a liquid lens) or provides additional custom defined post processing steps. Both, Acquisition Systems and Extensions, are dynamic libraries that can be loaded into OCTproZ during runtime.  

# 3. Processing Pipeline

Raw data, i.e. acquired spectral fringe pattern, from the OCT system is transferred to RAM until enough data for a user-defined amount of cross-sectional images, so-called B-scans, is acquired. Via direct memory access (DMA) this raw data batch is then copied asynchronously to GPU memory where OCT signal processing is executed. If the processed data needs to be stored or post processing steps are desired the processed OCT data can be transferred back to RAM with the use of DMA. An overview of the processing steps is depicted in Fig. \ref{fig:processing}.

 ![Processing pipeline of OCTproZ v1.1.0 Raw data from the OCT system is transferred to host RAM and via direct memory access (DMA) this raw data is then copied asynchronously to GPU memory where OCT signal processing is executed. If the processed data needs to be saved on the hard disk, it can be transferred back to host RAM using DMA. \label{fig:processing}](figures/processing_pipeline.png) 

A detailed description of each processing step can be found in the software repository. Here we just want to mention that the implementation of the fixed-pattern noise removal is based on a publication by Moon et al. [@moon2010reference] and the volume viewer is based on source code from an open source raycaster. [@pilia2018raycaster]
In order to avoid unnecessary data transfer to host memory, CUDA-OpenGL interoperability is used which allows the processed data to remain in GPU memory for visualization. 

# 4. Processing Performance 
Processing rate highly depends on the size of the raw data, the used computer hardware and resource usage by background or system processes. With common modern computer systems and typical data dimensions for OCT, OCTproZ achieves A-scan rates in the MHz range. Exemplary, table 1 shows two computer systems and their respective processing rates for the full processing pipeline. However, since the 3D live view is computationally intensive the processing rate changes noticeably depending on whether the volume viewer is activated or not. The used raw data set consists of 12 bit per sample, 1024 samples per line (corresponds to 512 samples per A-scan), 512 lines per frame and 256 frames per volume. As the volume is processed in batches, the batch size was set for each system to a reasonable number of B scans per buffer to avoid GPU memory overflow. It should be noted that this performance evaluation was done with OCTproZ v1.0.0 but is also valid for v1.1.0 if the newly introduced processing step for sinusoidal scan distortion correction is disabled.

<p style="text-align: center;"><small><b>Table 1</b>: Comparison of two computer systems and their respective processing rates for raw data sets with 12 bit per sample, 1024 samples per line, 512 lines per frame and 256 frames per volume.</small></p>

. |**Office Computer**|**Lab Computer**
:-----|:-----|:-----
CPU|Intel® Core i5-7500|AMD Ryzen Threadripper 1900X
RAM|16 GB|32 GB
GPU|NVIDIA Quadro K620|NVIDIA GeForce GTX 1080 Ti
Operating system|Windows 10|Ubuntu 16.04
B-scans per buffer|32|256
With 3D live view:| | 
  A-scans per second|~**$250 \cdot 10^{3}$**|~**$4.0 \cdot 10^{6}$**
  Volumes per second|~**$1.9$**|~**$30$**
Without 3D live view:| | 
   A-scans per second|~**$300 \cdot 10^{3}$**|~**$4.8 \cdot 10^{6}$**
   Volumes per second|~**$2.2$**|~**$36$**


# 5. Plug-in System

To develop custom plug-ins for OCTproZ and thus extends its functionality, a development kit is provided. It consists of a static library and a collection of C++ header files that specify which classes and methods have to be implemented to create custom plug-ins. Currently two kinds of plug-ins exist: Acquisition Systems and Extensions. For both we made examples including source code publicly available which may be used together with the open source and cross-platform integrated development environment Qt Creator as starting point for custom plug-in development.

For Acquisition System development the two key methods that need to be implemented are “startAcquisition()” and “stopAcquisition()”. In startAcquisition() an acquisition buffer needs to be filled and the corresponding boolean flag needs to be set. The processing thread in the main application continuously checks the acquisition buffer flag to transfer the acquired raw data to GPU as soon as the acquisition buffer is filled. When the acquisition is stopped, stopAcquisition() is called, where termination commands to stop hardware such as scanners may be implemented. 

Extensions have a wide area of use cases. As they are able to receive raw data and processed data via the Qt signals and slots mechanism, they are suitable for custom post-processing routines. The exact implementation of an Extension is mainly up to the developer and can also include hardware control. Therefore, Extensions are ideal for hardware control algorithms that rely on feedback from live OCT images. The best example of this is wavefront sensorless adaptive optics with a wavefront modulator such as a deformable mirror.  Particular care must be taken if high speed OCT systems are used, as the acquisition speed of OCT data may exceed the processing speed of the custom Extension. In this case, a routine within the Extension should be implemented that discards incoming OCT data if previous data is still processed. 

# 6. Conclusion
In this paper, we introduced OCTproZ, an open source software for live OCT signal processing. With the presented plug-in system, it is possible to develop software modules to use OCTproZ with custom OCT systems, thus reducing the OCT system development time significantly. OCTproZ is meant to be a collaborative project, where everyone involved in the field of OCT is invited to improve the software and share the changes within the community. 
We especially hope for more open source publications within the OCT community to reduce the time necessary for the replication of OCT processing algorithms and thereby accelerate scientific progress.

# Funding
This work was partially funded by the European Regional Development Fund (ERDF) and the state Lower Saxony as part of the project OPhonLas. 

![](figures/efre.png) 


# References

