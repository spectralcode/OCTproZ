 # <img style="vertical-align:middle" img src="images/octproz_icon.png" width="64"> OCTproZ - Processing Overview

This is a brief overview of the OCTproZ processing pipeline. </br></br>
OCT raw data from the OCT system is transferred to RAM
until a user-defined amount of B-scans is acquired (B-scans per buffer). Via direct memory access (DMA) this raw data batch is then copied asynchronously to GPU memory where OCT signal processing is performed.</br>

<p align="center">
  <img src="images/octproz_processingpipeline_linear.png">
</p>

Each box in the image above represents a CUDA kernel. To enhance processing performance some processing steps are combinend into a single kernel (e.g. k-linearization, dispersion compensation and windowing) to enhance processing performance. 


Processing Steps
--------

* **Data conversion**  </br>
Raw data (integer values with bit depth between 8 bit and 32 bit) is convertet to cuda compatible floating-point comlex data type cufftComplex.</br>

* **k-linearization**  </br>
Resamples OCT raw data evenly in k-space. You can define the resampling curve within the GUI by specifying coefficients of a third order polynomial and it is possible to choose between linear and 3rd order polynomial interpolation.

* **Dispersion compensation**  </br>
Numerical dispersion compenation. The k-linearized raw data is multiplied with a phase term ⅇ^(-ⅈθ(k) ) that cancels the phase shift introduced due dispersion mismatch in sample- and reference-arm. A user defined phase θ(k) can be specified in the GUI by providing the coefficients of a third order polynomial.</br>

* **Windowing**  </br>
The raw data is multiplied with a window function, which sets the signal to zero outside of a predefined interval. This reduces side lobes in the resulting signal after IFFT. The GUI allows to choose between different window functions (Gaussian, Hanning, Sine, Lanczos and Rectangular window) and to set their width and center position.</br>

* **IFFT**  </br>
Inverse Fourier transformation.</br>

* **Fixed-pattern noise removal**  </br>
Fixed pattern noise refers to structural artifacts in OCT images that appear as fixed horizontal lines. This step removes these artifacts and does not need a prerecorded reference signal.</br>

* **Truncate**  </br>
This step removes the mirror image of each B-scan which naturally occurs after IFFT.</br>

* **Logarithm**  </br>
Dynamic range compression (logarithm of magnitude) prepares the data for visualization.</br>

* **Backward scan correction**  </br>
To increase frame rate, a bidirectional scanning scheme can be used. However, this means that every other frame is flipped horizontally. The backward scan correction step unflippes these frames.  </br>

* **Sinusoidal scan correction**  </br>
A resonant scanner can be used for high speed OCT-systems. Due to the sinusoidal movement of the scanner the resulting B-scans would be distorted without this processing step. </br>

* **Visualization**  </br>
The processed data is displayed live in 1D (raw Data and A-scans), 2D (B-scans and En face view) and 3D.</br>
