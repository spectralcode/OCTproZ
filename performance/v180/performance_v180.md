 # <img style="vertical-align:middle" img src="../../images/octproz_icon.png" width="64"> OCTproZ - Performance Information

Processing rate highly depends on the size of the raw data, the used computer hardware and resource usage by background or system processes. With modern computer hardware and typical data dimensions for OCT, OCTproZ achieves A-scan rates in the MHz range.

A test data set with 12 bit per sample, 1024 samples per raw A-scan, 512 A-scans per B-scan and 256 B-scans per volume was used to measure the performance on different systems:

| |**Old Gaming Computer**|**NVIDIA Jetson Nano**|
|:-----|:-----|:-----|
|CPU|AMD Ryzen™ 5 1600|ARMv8 Processor rev 1 (v8l) x 4|
|RAM|16 GB|4 GB|
|GPU|NVIDIA GeForce GTX 1080|NVIDIA Tegra X1 (128-core Maxwell)|
|Operating system|Windows 10|Ubuntu 18.04 (JetPack 4.6.4)|
|A-scan rate with 3D view|~ 2.93 MHz (~ 22 volumes/s)|~ 94 kHz (~ 0.7 volumes/s)|
|A-scan rate without 3D view|~ 3.40 MHz (~ 26 volumes/s)|~ 181 kHz (~ 1.4 volumes/s)|


Please note that performance is highly dependent on the configured acquisition and processing parameters.

The following settings were used in the Virtual OCT System and OCTproZ v.1.8.0 during these tests:

| |**Old Gaming PC**|**NVIDIA Jetson Nano**|
|:-----|:-----|:-----|
|**Virtual OCT System Settings**| | |
|bit depth [bits]|12|12|
|Samples per raw A-scan|1024|1024|
|A-scan per B-scan|512|512|
|B-scans per buffer|256|32|
|Buffers per volume|1|8|
|Buffers to read from file|2|16|
|Wait after file read [us]|0|0|
|Copy file to RAM|enabled|disabled|
|Synchronize with processing|enabled|enabled|
|**OCTproZ Settings**| | |
|Bit shift sample values by 4|disabled|disabled|
|Sinusoidal scan correction|disabled|disabled|
|Flip every second B-scan|disabled|disabled|
|k-linearization|enabled|enabled|
|&emsp;interpolation|Cubic|Cubic|
|Dispersion Compensation|enabled|enabled|
|Windowing|enabled|enabled|
|Fixed-Pattern Noise Removal|enabled|enabled|
|B-scans for noise determination:|1|1|
|&emsp;once at start of measurement|enabled|enabled|
|&emsp;continuously|disabled|disabled|
|Log scaling|enabled|enabled|
|Post processing background |disabled|disabled|
|Stream Processed Data to Ram|disabled|enabled|
|**Volume Rendering Settings**| | |
|Mode|MIP|MIP|
|Ray Step|0.01|0.01|




How to Determine Performance
--------
 OCTproZ provides live performance information within the sidebar in the "Processing"-tab. Live performance estimation is performed and updated every 5 seconds:
<p align="center">
  <img src="20250504_performance_v180_gtx1080\20250504_octproz_v180_info_screenshot.png" >
</p>

For more in-depth profiling, you can use [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute). 
 
However, Nsight Compute does **not** support older GPUs, such as the GeForce GTX 1080 In such cases, you can use the now deprecated (but still usable) [NVIDIA Visual Profiler](https://developer.nvidia.com/nvidia-visual-profiler) 

The following screenshot from the NVIDIA Visual Profiler shows the profiler of OCTproZ v1.8.0 output using the old gaming computer setup:

 <p align="center">
  <img src="20250504_performance_v180_gtx1080\20250504_octproz_v180_old_gaming_pc_screenshot.png" >
</p>

The corresponding .nvvp file can be found [here](20250504_performance_v180_gtx1080\20250504_octproz_v180_old_gaming_pc.nvvp)
