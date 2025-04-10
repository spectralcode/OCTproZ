var __index = {"config":{"lang":["en"],"separator":"[\\s\\-]+","pipeline":["stopWordFilter"]},"docs":[{"location":"index.html","title":"Welcome to the OCTproZ v1.8.0 User Manual!","text":"<p>Date: April 10, 2025 \u2003 Author: Miroslav Zabic (zabicmagic@spectralcode.de)</p>"},{"location":"index.html#about","title":"About","text":"<p>OCTproZ is an open source and cross-platform software for optical coherence tomography (OCT) processing and visualization. A plug-in system enables the integration of custom OCT systems and software modules. You can find the most recent software release on the GitHub project page.</p>"},{"location":"index.html#user-guide","title":"User Guide","text":"<p>This is the user guide for OCTproZ. Browse by categories on the left, or if you are looking for a specific topic, use the search bar at the top of the page.</p> <p>Just like OCTproZ itself, this user guide is an ongoing project that undergoes continuous change. If you spot incomplete or incorrect information, feel free to email me or open an issue on the project page on GitHub.</p>"},{"location":"faq.html","title":"FAQ","text":""},{"location":"faq.html#which-oct-raw-data-format-is-supported-by-the-virtual-oct-system","title":"Which OCT raw data format is supported by the Virtual OCT System?","text":"<p>Raw data files that only contain the raw data are supported. The samples in the raw file must have a bit depth between 8 bits and 32 bits, the byte order must be little-endian and the raw data must be unpacked. For example, raw data with packed 12-bit samples (data for two samples is spread over 3 bytes) is currently not supported.</p> <p>If you have any questions, feel free to contact me: zabic@spectralcode.de</p>"},{"location":"functionality.html","title":"Core Functionality","text":"<p>Failure</p> <p>This part of the documentation is still work in progress.</p>"},{"location":"functionality.html#overview-of-the-graphical-user-interface","title":"Overview of the graphical user interface","text":""},{"location":"functionality.html#processing-tab","title":"Processing tab","text":""},{"location":"functionality.html#recording-tab","title":"Recording tab","text":""},{"location":"functionality.html#scheduled-recording","title":"Scheduled recording","text":""},{"location":"functionality.html#saving-and-loading-settings","title":"Saving and loading settings","text":""},{"location":"functionality.html#saving-and-loading-view","title":"Saving and loading view","text":""},{"location":"plugins.html","title":"Plugins","text":"<p>OCTproZ supports two distinct types of plugins:</p> Plugin Type Purpose Acquisition Systems Acquire raw data from OCT hardware or files Extensions Add new features and functionality to OCTproZ"},{"location":"plugins.html#available-plugins","title":"Available plugins","text":"<p>Acquisition Systems:</p> Name Description PhytoOCT A portable and low-cost OCT system. Virtual OCT System Can be used to load already acquired OCT raw data from the disk <p>Extensions:</p> Name Description Axial PSF Analyzer Measuring the FWHM of the axial point spread function. Camera Displays live view from a webcam. Demo Extension This demo extension is for developers. It has no useful functionality, but the code can be used as a template for developing custom extensions. Dispersion Estimator Helps determining suitable dispersion parameters d2 and d3 for better axial resolution. Image Statistics Displays useful image statistics, such as a histogram, in real time of currently acquired B-scans. Peak Detector Detects and displays the position of the highest peak in an A-scan. Phase Extraction Can be used to determine a suitable resampling curve for k-linearization. Signal Monitor Displays signal intensity. Useful during optical alignment for maximizing intensity on detector. Socket Stream Controlling OCTproZ remotely and streaming OCT data via TCP/IP, Websocket, IPC."},{"location":"plugins.html#custom-plugin-development","title":"Custom plugin development","text":"<p>Have a look at the plugin developer guide. </p>"},{"location":"processing.html","title":"Processing Pipeline","text":"<p>The OCT signal processing is entirely performed on the GPU. The spectral interferograms acquired by the OCT system are transferred to RAM until a user-defined number of B-scans is collected. This raw data batch is then copied to GPU memory for processing and display. Finally, the processed data is copied back to RAM, where it can be used for further analysis by plugins or saved to disk.</p> <p>The following image shows the processing pipeline. Although most processing steps are optional, data conversion, IFFT, and truncation are essential steps that can not be disabled.</p> OCTproZ processing pipeline"},{"location":"processing.html#processing-steps","title":"Processing Steps","text":""},{"location":"processing.html#data-conversion","title":"Data conversion","text":"<p>The first step of the OCT processing pipeline converts the incoming raw data, that may have a bit depth between 8 bit and 32 bit, to a single-precision, floating-point complex data type with a bit depth of 32 bit. This ensures that the processing pipeline can be executed for a variety of different input data types. Furthermore, a bit shift operation can be applied during the conversion process if necessary. Some digitizers, that are commonly used for swept-source OCT (SS-OCT), can be configured to use 16-bit integers to store 12-bit sample values in the most significant bits (e.g. ATS9373, Alazar Technologies Inc.). To extract the actual 12-bit value a right-shift by 4, which is equal to a division by 16, needs to be applied to every 16-bit integer.</p>"},{"location":"processing.html#dc-background-removal","title":"DC background removal","text":"<p>The DC component of the OCT signal is usually visible at the top of each B-scan as a bright line. To remove this DC component from each raw spectrum  \\( I_{\\mathrm{raw}}[m] \\), a rolling average with a user-adjustable window size \\( W \\) can be computed and subtracted from the raw signal. Ignoring boundary checks, the DC\u2011corrected signal is calculated by</p> \\[     I_{\\mathrm{ac}}[m] = I_{\\mathrm{raw}}[m] - \\frac{1}{2W} \\sum_{n=m-W+1}^{m+W} I_{\\mathrm{raw}}[n]. \\] <p>In cases where the DC component does not vary significantly over time, the fixed-pattern noise removal step can also eliminate it. </p>"},{"location":"processing.html#k-linearization","title":"k-linearization","text":"<p>To convert the acquired raw OCT data into a depth profile, an inverse Fourier transform can be used, which relates wavenumber k to physical distance. For optimal axial resolution, the raw data must be uniformly sampled in k-space. However, depending on the used hardware setup, the acquired spectral fringe pattern is usually not linear in k. In swept-source OCT, the wavenumber-time characteristics of the light source are often non-linear. In spectrometer-based OCT systems, the exact pixel position to wavenumber relationship depends on the optical elements of the spectrometer. There are hardware based approaches that enable a k-linear sampling like using a k-clock for non-linear temporal sampling of the raw signal in swept-source OCT or using special k-linear spectrometer designs.</p> <p>Alternatively, there are software-based approaches to generate a signal with a k-space uniform sampling by resampling the data such that the data points are evenly distributed in k-space. This process is known as k-linearization. For the software-based approach in OCTproZ, a user-defined resampling curve \\( r[m] \\) can be specified by providing the coefficients of a third-order polynomial. The resampling curve is a lookup table that assigns every index \\( m \\) of the raw data array \\( I_{\\mathrm{raw}}[m] \\) an index \\( m^\\prime \\), i.e. \\( m^\\prime = r[m] \\). To obtain a k-linearized raw data array \\( I_{\\mathrm{k}}[m] \\), the sample value at the index \\( m^\\prime \\) needs to be interpolated and remapped to the array position with index \\( m \\). The simplest way to do this is by using linear interpolation:</p> \\[     I_{\\mathrm{k}}[m]      = I_{\\mathrm{raw}}[\\lfloor m^\\prime \\rfloor]      + \\bigl(m^\\prime - \\lfloor m^\\prime \\rfloor \\bigr)        \\bigl(I_{\\mathrm{raw}}[\\lfloor m^\\prime \\rfloor+1] - I_{\\mathrm{raw}}[\\lfloor m^\\prime \\rfloor]\\bigr) \\] <p>\\( \\lfloor x \\rfloor \\) denotes the floor function that takes as input \\( x \\), a real number, and gives as output the greatest integer less than or equal to \\( x \\).</p> <p>Note</p> <p>The polynomial used for the resampling curve has the form: r(m) = c<sub>0</sub> + (c<sub>1</sub>/N)m + (c<sub>2</sub>/N<sup>2</sup>)m<sup>2</sup> + (c<sub>3</sub>/N<sup>3</sup>)m<sup>3</sup> where N is the number of samples per raw A-scan - 1.</p> <p>Note</p> <p>A custom resampling curve (that does not need to be a polynomial fit) can be loaded by clicking on Extras \u2192 Resampling curve for k-linearization \u2192 Load custom curve from file... The structure of the csv file with the curve data should be the same as the structure of the csv file that you get by right clicking on the resampling curve plot in the sidebar and saving the plot as csv file.</p> <p>Note</p> <p>You can use the PhaseExtractionExtension for easy resampling curve determination.</p> <p>Currently, three interpolation methods are available in OCTproZ: Linear, Cubic Spline (Catmull-Rom Spline), and Lanczos. These methods represent a trade-off between speed and accuracy, with Linear being the fastest and Lanczos being the most accurate. The figure below shows typical interpolation artifacts that can be seen when using the different interpolation methods:</p> Image artifacts of different interpolation methods for k-linearization indicated by red arrows"},{"location":"processing.html#dispersion-compensation","title":"Dispersion compensation","text":"<p>Differences in the dispersive media lengths in the sample and reference arms of an OCT system introduce a wavenumber\u2011dependent phase shift on the raw signal that degrades axial resolution of the processed OCT images. While hardware\u2011based methods (e.g. prism pairs used as variable\u2011thickness windows) can physically balance dispersion mismatch, numerical dispersion compensation offers a flexible alternative, especially in cases where the mismatch is mainly caused by the sample and not only by the optical setup itself.</p> <p>In numerical dispersion compensation, the raw signal \\( I_{\\mathrm{raw}}[m] \\) is corrected by multiplying it with a phase term that cancels the dispersion\u2011induced phase shift:</p> \\[     I_{\\mathrm{comp}}[m] = I_{\\mathrm{raw}}[m] \\cdot e^{-i\\,\\theta\\bigl(k\\bigr)}, \\] <p>where \\( \\theta(k) \\) is a user\u2011defined phase function. Expressing the phase term in its real and imaginary parts as</p> \\[     e^{-i\\,\\theta\\bigl(k\\bigr)} = \\cos\\bigl(\\theta(k)\\bigr) - i\\,\\sin\\bigl(\\theta(k)\\bigr) \\] <p>and noting that the imaginary part of \\( I_{\\mathrm{raw}}[m] \\) is zero, the multiplication simplifies to</p> \\[     \\begin{aligned}         \\mathrm{Re}\\{ I_{\\mathrm{comp}}[m] \\} &amp;= I_{\\mathrm{raw}}[m] \\cdot \\cos\\bigl(\\theta(k)\\bigr),\\\\[1ex]         \\mathrm{Im}\\{ I_{\\mathrm{comp}}[m] \\} &amp;= -\\,I_{\\mathrm{raw}}[m] \\cdot \\sin\\bigl(\\theta(k)\\bigr).     \\end{aligned} \\] <p>This simplified calculation avoids performing a full complex multiplication. This appraoch is slightly different from the one described by Maciej Wojtkowski et al. (2004). In their method, the Hilbert transform is used to generate the imaginary part of the raw signal, and then a full complex multiplication is performed. Both approaches result in identical OCT data.</p> <p>The phase function \\( \\theta(k) \\) is implemented as a third\u2011order polynomial with coefficients \\( d_0, d_1, d_2, d_3 \\), which can be set in the GUI. \\( d_0 \\) does not change the resulting OCT images at all; \\( d_1 \\) can be used to shift the OCT image in axial direction without altering the axial resolution. Only \\( d_2 \\) and \\( d_3 \\) are responsible for the actual dispersion compensation that can improve axial resolution. </p> <p>In many cases, it is possible to find good values for \\( d_2 \\) and \\( d_3 \\) by manually varying them  until the resulting image shows the best axial resolution. However, it is also possible to use the dispersion estimator extension to obtain suitable coefficient values by optimizing an A-scan quality metric. </p>"},{"location":"processing.html#windowing","title":"Windowing","text":"<p>The raw data is multiplied by a window function, which sets the signal to zero outside of a predefined interval. This reduces side lobes in the resulting signal after IFFT. The GUI allows to choose between different window functions (Gaussian, Hanning, Sine, Lanczos and Rectangular window) and to set their width and center position.</p>"},{"location":"processing.html#inverse-fast-fourier-transform-ifft","title":"Inverse Fast Fourier Transform (IFFT)","text":"<p>The inverse Fourier transform is the essential processing step to calculate the depth profile from a spectral interferograms. The IFFT output is normalized by dividing each sample by the total number of samples.</p>"},{"location":"processing.html#fixed-pattern-noise-removal","title":"Fixed-pattern noise removal","text":"<p>Fixed pattern noise refers to structural artifacts in OCT images that appear as fixed horizontal lines. These artifacts are caused, for example, by variations in pixel response in the CCD camera in spectrometer based OCT systems or spurious etalons within the optical OCT setup. A common approach to reduce fixed pattern noise is to acquire a reference signal in absence of a sample and subtract it from all subsequent recordings. In OCTproZ, the minimum-variance mean-line subtraction method that was described by Moon et al. (2010) can be used. This approach does not require an additional reference recording and can be applied continuously such that fixed pattern noise due spectral intensity variation of the source is reduced as well.</p>"},{"location":"processing.html#truncate","title":"Truncate","text":"<p>This step removes the mirror image on the opposite side of zero pathlength by cropping half of the processed OCT data. The mirror image is sometimes referred to as mirror artifact or complex conjugate artifact. It originates from the fact that the inverse Fourier transform is applied to a real-valued signal which results in a conjugate symmetric signal (i.e. the positive and negative distances are complex conjugates of each other).</p>"},{"location":"processing.html#logarithm-and-dynamic-range-adjustment","title":"Logarithm and dynamic range adjustment","text":"<p>For better visualization of OCT data, some form of dynamic range compression is usually used because the smallest and largest signals can differ by several orders of magnitude. The most common method, which is also used in OCTproZ, is simply to take the logarithm of the signal after IFFT:</p> \\[ i[z] = 20 \\, \\log_{10}\\!\\left|\\mathcal{F}^{-1}\\{I_k[m]\\}\\right| \\] <p>In addition, dynamic range adjustment is performed to enable the user to set minimum and maximum values (in dB) that should be displayed:</p> \\[ i_{\\text{adj}}[z] = \\text{coeff} \\left( \\frac{i[z]- \\text{min}}{\\text{max} - \\text{min}} + \\text{addend} \\right), \\] <p>The parameters coeff, min, max, and addend can be set by the user. Usually, min and max are chosen so that the noise floor in the OCT images appears quite dark and the actual signal of interest appears bright. Coeff can be used to adjust the contrast of the image, and addend can be used to adjust the brightness. Typically, these values are set to coeff = 1 and addend = 0.</p>"},{"location":"processing.html#backward-scan-correction","title":"Backward scan correction","text":"<p>To increase frame rate, a bidirectional scanning scheme can be used. However, this means that every other frame is flipped. The backward scan correction step unflips these frames.</p> Bidirectional scanning scheme and the effect of backward scan correction on the en face view <p>The image above shows the effect of the backward scan correction on the en face view of an OCT volume that was acquired using a bidirectional scanning scheme. A piece of wood with a laser burned hole was used as sample. Left: Spot path on sample when a bidirectional scanning scheme is applied. Middle: En face view with enabled backward scan correction. Right: En face view when backward scan correction is disabled.</p>"},{"location":"processing.html#sinusoidal-scan-correction","title":"Sinusoidal scan correction","text":"<p>A resonant scanner can be used for high-speed OCT systems. Due to the sinusoidal scan motion, the resulting images are distorted. This processing step uses linear interpolation to unstretch this sinusoidal distortion. The figure below shows the effect of this processing step.</p> En face view of a grid structure acquired with sinusoidal scanning. Left: sinusoidal scann correction not applied, distortions visible. Right: en face view after applying sinusoidal scan correction,"},{"location":"processing.html#effect-of-single-processing-steps","title":"Effect of single processing steps","text":"<p>To illustrate the effect of single processing steps, B-scans of an OCT phantom (APL-OP01, Arden Photonics, UK) were acquired with a custom made SS-OCT system without k-klocking and with a slight dispersion imbalance. The acquired raw data was processed multiple times, each time with a different processing step disabled:</p> The B-scans show a test pattern of an OCT phantom (APL-OP01, Arden Photonics, UK). Below each B-scan is an enlarged view of a corresponding area framed in red within the B-scan. a) The full processing pipeline is enabled. b) k linearization is disabled (all other steps are enabled). c) Dispersion compensation is disabled (all other steps are enabled). d) Windowing is disabled (all other steps are enabled). e) Fixed-pattern noise removal is disabled (all other steps are enabled). The red arrows point to horizontal structural artifacts that are visible if fixed-pattern noise removal is disabled."},{"location":"quickstart.html","title":"Quick Start Guide","text":"<p>This section shows you how to load an OCT raw dataset with the Virtual OCT System Extension that is provided with OCTproZ. For testing purposes you can download a test data set from here.</p>"},{"location":"quickstart.html#1-open-virtual-oct-system","title":"1. Open Virtual OCT System","text":"<p>Click on File \u2192 Open System</p> <p></p> <p>The system manager opens in a new window. Select \"Virtual OCT System\" and click on \"Select\"</p> <p></p>"},{"location":"quickstart.html#2-set-virtual-oct-system-settings","title":"2. Set Virtual OCT System settings","text":"<p>Click on File \u2192 System Settings</p> <p></p> <p>The system settings window opens. Click on Select file and select the OCT raw data file you want to open. Enter the parameters in the settings window according the dimensions of your raw data (bit depth per sample, samples per line,...). For more information on the individual parameters, click on the question mark in the upper right corner and then on the input field you would like to learn more about.</p> <p></p>"},{"location":"quickstart.html#3-set-processing-parameters-in-sidebar","title":"3. Set processing parameters in sidebar","text":"<p>Enter suitable processing parameters in the sidebar. The white curves in the k-linearization, dispersion compensation and windowing plots are reference curves that indicate how a curve would look like that does not effect the processing result at all. In other words: If your curve looks exactly as the white curve then the processing result will not change if this particular processing step is deactivated. For more information on processing, see the processing pipeline section.</p> <p></p>"},{"location":"quickstart.html#4-start-the-processing","title":"4. Start the processing","text":"<p>Click on the \"Start\" button in the top left of the sidebar.</p> <p></p>"},{"location":"quickstart.html#5-adjust-display-settings","title":"5. Adjust display settings","text":"<p>Hover your mouse over one of the output windows and a control panel will appear that you can use to adjust the display settings.</p> <p></p>"},{"location":"troubleshooting.html","title":"Troubleshooting","text":""},{"location":"troubleshooting.html#no-visual-output-b-scan-en-face-view-and-volume-windows-are-black-after-clicking-start-button","title":"No visual output. B-scan, En Face View and Volume windows are black after clicking start button","text":"<ul> <li>Check if you have a CUDA compatible GPU.</li> <li>Check if your monitor cable is connected to the GPU. If your monitor is connected to the motherboard, the processing will still run on the GPU but there will be no visual output in the OpenGL windows.</li> <li>Check if you have the right processing settings. With some settings, the complete output is set to 0 and the output windows remain black. For example if all k-linearization coefficients are 0, the output will be 0. If the windowing fill factor is 0, the output will be 0. If the grayscale conversion multiplicator is 0, the output will be 0.</li> <li>Check if the stretch parameters are greater than 0 in your display settings. See step 5 in the quick start guide</li> <li>If you are using Windows Remote Desktop, OpenGL may not work properly which can cause black output windows.</li> </ul>"},{"location":"troubleshooting.html#crash-right-after-clicking-start-button-and-using-virtual-oct-system","title":"Crash right after clicking start button and using Virtual OCT System","text":"<ul> <li>Maybe the size of the OCT data buffer is too large and you are running out of GPU memory. Try reducing the buffer size by reducing B-scans per buffer in the Virtual OCT System settings.</li> </ul>"},{"location":"visualization.html","title":"Visualization","text":"<p>For live visualization of the processed data in 2D and 3D, the user has access to three different output windows: B-scan, en face view and volume. B-scan and en face view are orthogonal cross-sectional slices of the volume, which can be maximum intensity projections or averaged layers of a user-defined amount of layers of the volume. For easier orientation, red marker lines can be overlaid to indicate the current B-scan slice position within the en face view and vice versa.</p> <p>The interactive volume viewer displays acquired OCT volumes without cropping or downsampling in real time. As soon as one batch of data is processed, the corresponding part of the volume is updated and rendered. In order to avoid unnecessary data transfer to host memory, CUDA-OpenGL interoperability is used which allows the processed data to remain in GPU memory for visualization.</p>"},{"location":"visualization.html#volume-rendering","title":"Volume rendering","text":"<p>Here are some example images showcasing an OCT volume of a fingernail rendered using the implemented volume rendering techniques:</p> <p></p>"}]}