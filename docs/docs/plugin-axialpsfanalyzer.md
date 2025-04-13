# Axial PSF Analyzer Extension

The Axial PSF Analyzer can be used to measure the axial resolution of an OCT system, defined as the Full Width at Half Maximum (FWHM) of the axial Point Spread Function (PSF).

To perform the measurement, place a mirror as the sample and set the Region of Interest (ROI) so that it covers the horizontal line in the B-scan corresponding to the mirror surface. All A-scans within the selected ROI are averaged and displayed in the 1D plot on the right side. A Gaussian function is then fitted to this averaged A-scan to estimate the FWHM.


<figure markdown="span">
	![Axial PSF Analyzer Screenshot](images/plugins/axialpsfanalyzer_screenshot.png)
	<figcaption>Axial PSF Analyzer Extension Interface</figcaption>
</figure>

## How to use

!!! note
	For a correct measurement, make sure to disable logarithmic scaling in the Processing tab of the OCTproZ sidebar. The Axial PSF Analyzer currently only supports Gaussian fits to linearly scaled A-scans.

## User interface
| Parameter | Description |
|-----------|-------------|
| Buffer |  The buffer number from which you want to grab the frame. If you only use one buffer per volume, or if it does not matter which specific frame is used for the estimation, select All. This will grab the frame from the next available buffer. |
| Frame | The frame number within the buffer. Together with the buffer number, this allows you to select the specific frame within the OCT volume that should be grabbed. If you only use one frame per buffer, or if it does not matter which specific frame is used for the estimation, select 0. |
| Auto fetch every nth buffer | Automatically updates the OCT data every n-th buffer. Useful during live measurements, for example when moving the mirror through the entire axial imaging range to record the roll-off. |
| Fit model | currently not used |
| Autoscaling | If enabled, the A-scan plot with the Gaussian fit is automatically scaled so that the entire A-scan is visible. |