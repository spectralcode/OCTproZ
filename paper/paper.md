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

Optical coherence tomography (OCT) is a non-invasive imaging technique used primarily in the medical field, especially in ophthalmology. Current research in OCT includes hardware as well software aspects. Both areas require a software solution for processing of the acquired raw data. 

Here we present an open source software for OCT processing with an easy to use graphical user interface. The implemented OCT processing pipeline enables A-scan processing rates in the MHz range. Custom OCT systems, or any other source of Fourier Domain OCT raw data, can be integrated via a developed plug-in system, which also allows the development of custom post processing modules.

# 1. Introduction

Optical coherence tomography (OCT) is a non-invasive imaging technique used primarily in the medical field, especially in ophthalmology. The core element of any OCT system is an optical interferometer that generates a spectral fringe pattern by combining reference beam and the backscattered light from a sample. To obtain an interpretable image from this acquired raw OCT signal several processing steps are necessary, whereby the inverse Fourier transform represents an essential step. As the possible acquisition speed for raw OCT data has increased constantly, more sophisticated methods were needed for processing and live visualization of the acquired OCT data. A particularly impressive setup was presented by Choi et al. [@choi2012spectral] that utilizes twenty FPGA-modules for real-time OCT signal processing and a graphics processing unit (GPU) for volume rendering. Nowadays, processing is typically done on graphics cards [@zhang2010real], not FPGAs, because implementing algorithms on GPUs is more flexible and takes less time. [@li2011scalable] Over the years, many different processing techniques have been published that improve axial [@wojtkowski2004ultrahigh,@cense2004ultrahigh] and lateral resolution [@yu2007improved,@shen2017improving], enhance imaging depth [@moiseev2012digital] and extend the imaging functionality itself. [@park2003real] Most of these publications do not provide the actual software implementation, i.e. the source code, of the proposed processing technique.  A commendable exemption is the GPU accelerated OCT processing pipeline published by Jian et al. [@jian2013graphics] The associated source code, which demonstrates an implementation of OCT data processing and visualization and does not include any advanced features such a graphical user interface (GUI), already consists of several thousand lines. Thus, the most time consuming task of Fourier Domain OCT (FD-OCT) system development is not the optical setup, but the software development. The software can be separated into hardware control and signal processing, whereby the former being a highly individual, hardware-dependent software module and the latter being a generic software module, which is almost identical for many systems. To drastically reduce OCT system development time, we present OCTproZ, an open source OCT processing software that can easily be extended, via a plug-in system, for many different hardware setups. In this paper we describe the key functionality and structure of the software at the current release (version 1.0).

# 2. Basic overview of OCTproZ

OCTproZ performs live signal processing and visualization of OCT data. It is written in C++, uses the cross-platform application framework Qt for the GUI and utilizes Nvidia’s computer unified device architecture (CUDA) for GPU parallel computing. A screenshot of the application can be seen in Fig. \ref{fig:screenshot}

 ![Screenshot of OCTproZ v1.0. Processing settings visible in the left panel can be changed before processing is started or while processing is in progress. Processed data is live visualized in 2D as cross sectional images (B-scan and en face view) and in 3D as interactive volume rendering. The live view shows a piece of wood with a couple layers of tape and a laser burned hole. \label{fig:screenshot}](figures/20191122_screenshot3d.png)


The software can be separated into three parts: main application, development kit (DevKit) and plug-ins. The main application, OCTproZ itself, contains the logic for GUI, processing and visualization. The DevKit, which is implemented as static library, provides the necessary interface for plugin development. Plug-ins can be one of two kinds: “Acquisition Systems” or “Extensions”. The former represent software implementations of physical or virtual OCT system hardware, the later are software modules that extend the functionality of an OCT system (e.g. software control of a liquid lens) or provides additional custom defined post processing steps. Both, Acquisition Systems and Extensions, are dynamic libraries that can be loaded into OCTproZ during runtime.  

# 3. Processing Pipeline

Raw data, i.e. acquired spectral fringe pattern, from the OCT system is transferred to RAM until enough data for a user-defined amount of cross-sectional images, so-called B-scans, is acquired. Via direct memory access (DMA) this raw data batch is then copied asynchronously to GPU memory where OCT signal processing is executed. If the processed data needs to be stored or post processing steps are desired the processed OCT data can be transferred back to RAM with the use of DMA. An overview of the processing steps is depicted in Fig. \ref{fig:processing}. The processing steps are in detail the following:

 ![Processing pipeline of OCTproZ v1.0. Raw data from the OCT system is transferred to host RAM and via direct memory access (DMA) this raw data is then copied asynchronously to GPU memory where OCT signal processing is executed. If the processed data needs to be saved on the hard disk, it can be transferred back to host RAM using DMA. \label{fig:processing}](figures/processing_pipeline.png) 


**Data conversion:**
The first step of the OCT processing pipeline converts the incoming raw data, that may have a bit depth between 8 bit and 32 bit, to a single-precision, floating-point complex data type with a bit depth of 32 bit. This ensures that the processing pipeline can be executed for a variety of different input data types. Furthermore, a bit shift operation is applied during the conversion process if necessary. Some digitizers, that are commonly used for swept source OCT (SS-OCT), can be configured to use 16-bit integers to store 12-bit sample values in the most significant bits (e.g. ATS9373, Alazar Technologies Inc.). To extract the actual 12-bit value a right-shift by 4, which is equal to a division by 16, needs to be applied to every 16-bit integer. 


**k linearization:**
To convert the acquired raw OCT data into a depth profile, inverse Fourier transform is used, which relates wavenumber k and physical distance. Depending on the used hardware setup, the acquired spectral fringe pattern is usually not linear in k. In SS-OCT the temporal sampling points do not necessarily have to be spaced in k domain evenly, especially if k clocking is not used and in spectrometer based FD-OCT systems the interference signal is usually acquired linear in wavelength. The k-linearization resamples the raw data evenly in k space, which improves axial resolution. 
A user defined resampling curve $r[j]$ can be specified by providing the coefficients of a third order polynomial. The resampling curve is a look up table that assigns every index $j$ of the raw data array $S_{raw}[j]$ an index $j'$, i.e. $j'=r[j]$ . To obtain a k-linearized raw data array $S_{k}[j]$, the value at the index $j'$ needs to be interpolated and remapped to the array position with index $j$. In the current version of OCTproZ the user can choose between linear and 3rd order polynomial interpolation for this task. \autoref{eq:linearinterpolation} describes the k-linearization with linear interpolation; $\lfloor x \rfloor$ denotes the floor function that takes as input $x$ a real number and gives as output the greatest integer less than or equal to $x$. 

\begin{equation}\label{eq:linearinterpolation}
S_k[j] = S_{raw}[\lfloor j' \rfloor] + (j' - \lfloor j' \rfloor )(S_{raw} [ \lfloor j' \rfloor +1]-S_{raw} [ \lfloor j' \rfloor ])
\end{equation}


**Dispersion compensation:**
If sample and reference arm of an OCT system contain different length of dispersive media, a wavenumber dependent phase shift is introduced to the signal and axial resolution decreases. Such dispersion mismatch usually occurs when the length or the number of optical fiber components and lenses is not identical in both optical arms. In this case, a hardware based dispersion compensation, such as variable-thickness fused-silica and BK7 prisms within in the sample arm [@drexler1999vivo], can be applied. A more convenient way to compensate for the additional phase shift, especially if the dispersion mismatch is introduced mainly by the sample itself, is numerical dispersion compensation. Hereby the signal is multiplied with a phase term $e^{(-i \Theta (k))}$ that exactly cancels the phase shift introduced due dispersion mismatch. [@cense2004ultrahigh] A user defined phase $\Theta (k)$ can be specified in the GUI by providing the coefficients of a third order polynomial. 


**Windowing:**
Windowing is a basic step in digital signal processing that is applied right before the Fourier transform to reduce side lobes in the resulting signal. It is in essence a multiplication of the k linear interference signal with a window function, which sets the signal to zero outside of a predefined interval. The GUI allows to choose between different window functions (Gaussian, Hanning, Sine, Lanczos and Rectangular window) and to set their width and center position.


**IFFT:**
The inverse Fourier transformation is the essential processing step to calculate the depth profile from the acquired and pre-processed (k-lineariziation, dispersion compensation, windowing) fringe pattern. OCTproZ utilizes the NVIDIA CUDA Fast Fourier Transformation library (cuFFT) to execute the inverse Fast Fourier Transform (IFFT).


**Fixed-pattern noise removal:**
Fixed pattern noise refers to structural artifacts in OCT images that appear as fixed horizontal lines. These artifacts are caused, for example, by variations in pixel response in the CCD camera in spectrometer based OCT systems or spurious etalons within the optical OCT setup. [@de2008spectral] A common approach to reduce fixed pattern noise is to acquire a reference signal in absence of a sample and subtract it from all subsequent recordings. In OCTproZ we have implemented the minimum-variance mean-line subtraction method that was described by Moon et al. [@moon2010reference] This approach does not require an additional reference recording and can be applied continuously such that fixed pattern noise due spectral intensity variation of the source is reduced as well. 


**Truncate and logarithm:**
Since the inverse Fourier transformation is applied to a real-valued signal, the result must be Hermitian symmetric. As consequence, mirror images, with the zero path reference as the mirror axis, can be seen in the resulting B-scans. If numerical dispersion correction is applied prior IFFT these mirror images are not identical with their counterparts but appear blurred. To avoid displaying duplicate or blurred images and reduce data size the data is truncated. The computation of magnitude and dynamic range compression (logarithm of magnitude) to prepare the data for visualization is done in the truncate processing step as well. 


**Backward scan correction:**
To increase frame rate, a bidirectional scanning scheme can be used. [@wieser2014high,@kolb2019live] However, this means that every other frame is flipped horizontally. The backward scan correction step unflippes these frames. Fig. \ref{fig:bscanflip} shows the effects of this processing step on the en face view of a volume that was acquired using a bidirectional scanning scheme.

 ![Effect of the backward scan correction on the en face view of an OCT volume that was acquired using a bidirectional scanning scheme. A piece of wood with a laser burned hole was used as sample. Left: Spot path on sample when a bidirectional scanning scheme is applied. Middle: En face view with enabled backward scan correction. Right: En face view when backward scan correction is disabled. \label{fig:bscanflip}](figures/bscanflip_overview_text_small.png) 
 

**Visualization:**
For live visualization of the processed data in 2D and 3D, the user has access to three different output windows: B-scan, en face view and volume. B-scan and en face view are orthogonal cross-sectional slices of the volume, which can be maximum intensity projections or averaged layers of a user-defined amount of layers of the volume. For easier orientation red marker lines can be overlaid to indicate the current B-scan slice position within the en face view and vice versa.  
The interactive volume viewer displays acquired OCT volumes without cropping or downsampling in real time. As soon as one batch of data is processed, the corresponding part of the volume is updated and rendered with maximum intensity projection, alpha blending or isosurfaces. The volume viewer is based on source code from an open source raycaster. [@raycaster]
In order to avoid unnecessary data transfer to host memory, CUDA-OpenGL interoperability is used which allows the processed data to remain in GPU memory for visualization. 
However, it is possible to transfer the data to the host memory to save it on the hard disk, display individual axial depth profiles, so-called A-scans, in a 1D plot or use it within custom Extensions.

Every processing step, except data conversion and IFFT, can be enabled and disabled during processing. To illustrate the effect of singe processing steps, B-scans of an OCT phantom (APL-OP01, Arden Photonics, UK) were acquired with a custom made SS-OCT system without k-klocking and with a slight dispersion imbalance. The acquired raw data was processed multiple times, each time with a different processing step disabled, see Fig. \ref{fig:effectofprocessing}. 

 ![Effect of disabling single processing steps on resulting B-scan. The B-scans show a test pattern of an OCT phantom (APL-OP01, Arden Photonics, UK). Below each B-scan is an enlarged view of the corresponding area framed in red within the B-scan. a) The full processing pipeline, as described in section 3, is enabled. b) k linearization is disabled (all other steps are enabled). c) Dispersion compensation is disabled (all other steps are enabled). d) Windowing is disabled (all other steps are enabled). e) Fixed-pattern noise removal is disabled (all other steps are enabled). The red arrows point to horizontal structural artifacts that are visible if fixed-pattern noise removal is disabled. \label{fig:effectofprocessing}](figures/overview.png) 

# 4. Processing Performance 
Processing rate highly depends on the size of the raw data, the used computer hardware and resource usage by background or system processes. With common modern computer systems and typical data dimensions for OCT, OCTproZ achieves A-scan rates in the MHz range. Exemplary, table 1 shows two computer systems and their respective processing rates for the full processing pipeline. However, since the 3D live view is computationally intensive the processing rate changes noticeably depending on whether the volume viewer is activated or not. The used raw data set consists of 12 bit per sample, 1024 samples per line (corresponds to 512 samples per A-scan), 512 lines per frame and 256 frames per volume. As the volume is processed in batches, the batch size was set for each system to a reasonable number of B scans per buffer to avoid GPU memory overflow. 

Table 1: Comparison of two computer systems and their respective processing rates for raw data sets with 12 bit per sample, 1024 samples per line, 512 lines per frame and 256 frames per volume.

 
. |**Office Computer**|**Lab Computer**
:-----|:-----|:-----
CPU|Intel® Core i5-7500|AMD Ryzen Threadripper 1900X
RAM|16 GB|32 GB
GPU|NVIDIA Quadro K620|NVIDIA GeForce GTX 1080 Ti
Operating system|Windows 10|Ubuntu 16.04
B-scans per buffer|32|256
With 3D live view:| | 
  A-scans per second|**~ 250⋅10<sup>3</sup>**|**~ 4.0⋅10<sup>6</sup>**
  Volumes per second|**~ 1.9**|**~ 30**
Without 3D live view:| | 
   A-scans per second|**~ 300⋅10<sup>3</sup>**|**~ 4.8⋅10<sup>36</sup>**
   Volumes per Second|**~ 2.2**|**~ 36**

