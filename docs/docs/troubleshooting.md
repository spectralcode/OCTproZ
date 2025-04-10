# Troubleshooting

### No visual output. B-scan, En Face View and Volume windows are black after clicking start button

- Check if you have a CUDA compatible GPU.
- Check if your monitor cable is connected to the GPU. If your monitor is connected to the motherboard, the processing will still run on the GPU but there will be no visual output in the OpenGL windows.
- Check if you have the right processing settings. With some settings, the complete output is set to 0 and the output windows remain black. For example if all k-linearization coefficients are 0, the output will be 0. If the windowing fill factor is 0, the output will be 0. If the grayscale conversion multiplicator is 0, the output will be 0.
- Check if the stretch parameters are greater than 0 in your display settings. See step 5 in the [quick start guide](quickstart.md)
- If you are using Windows Remote Desktop, OpenGL may not work properly which can cause black output windows.

### Crash right after clicking start button and using Virtual OCT System

- Maybe the size of the OCT data buffer is too large and you are running out of GPU memory. Try reducing the buffer size by reducing *B-scans per buffer* in the Virtual OCT System settings.
