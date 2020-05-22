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

