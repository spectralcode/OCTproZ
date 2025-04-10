# Plugins

OCTproZ supports two distinct types of plugins:

| **Plugin Type** | **Purpose** |
|-----------------|-------------|
| Acquisition Systems | Acquire raw data from OCT hardware or files | 
| Extensions | Add new features and functionality to OCTproZ |

## Available plugins


__Acquisition Systems:__

| Name | Description |
|-----|-----|
|[PhytoOCT](https://github.com/spectralcode/PhytoOCT)| A portable and low-cost OCT system.|
|[Virtual OCT System](octproz_project/octproz_plugins/octproz_virtual_oct_system)| Can be used to load already acquired OCT raw data from the disk|


__Extensions:__

| Name | Description |
|------|-------------|
|[Axial PSF Analyzer](https://github.com/spectralcode/AxialPsfAnalyzerExtension)| Measuring the FWHM of the axial point spread function.|
|[Camera](https://github.com/spectralcode/CameraExtension)| Displays live view from a webcam.|
|[Demo Extension](octproz_project/octproz_plugins/octproz_demo_extension)| This demo extension is for developers. It has no useful functionality, but the code can be used as a template for developing custom extensions.|
|[Dispersion Estimator](https://github.com/spectralcode/DispersionEstimatorExtension)| Helps determining suitable dispersion parameters d2 and d3 for better axial resolution. |
|[Image Statistics](https://github.com/spectralcode/ImageStatisticsExtension)| Displays useful image statistics, such as a histogram, in real time of currently acquired B-scans. |
|[Peak Detector](https://github.com/spectralcode/PeakDetectorExtension)| Detects and displays the position of the highest peak in an A-scan.|
|[Phase Extraction](https://github.com/spectralcode/PhaseExtractionExtension)| Can be used to determine a suitable resampling curve for k-linearization.|
|[Signal Monitor](https://github.com/spectralcode/SignalMonitorExtension)| Displays signal intensity. Useful during optical alignment for maximizing intensity on detector.|
|[Socket Stream](https://github.com/spectralcode/SocketStreamExtension)| Controlling OCTproZ remotely and streaming OCT data via TCP/IP, Websocket, IPC.|

## Custom plugin development
Have a look at the [plugin developer guide](https://spectralcode.github.io/OCTproZ/site/developer.html). 