# FAQ

## Which OCT raw data format is supported by the Virtual OCT System?

Raw data files that only contain the raw data are supported. The samples in the raw file must have a bit depth between 8 bits and 32 bits, the byte order must be little-endian and the raw data must be unpacked. For example, raw data with packed 12-bit samples (data for two samples is spread over 3 bytes) is currently not supported.

If you have any questions, feel free to contact me: zabic@spectralcode.de
