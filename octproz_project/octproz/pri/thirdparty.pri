#FFTW - copy dlls to octproz.exe folder (windows only)
win32 {
	FFTW_DLL_SRC = $$shell_path($$PWD/../../thirdparty/fftw)

	CONFIG(debug, debug|release) {
		FFTW_DLL_DST = $$shell_path($$OUT_PWD/debug)
	} else {
		FFTW_DLL_DST = $$shell_path($$OUT_PWD/release)
	}

	QMAKE_POST_LINK += $$quote(if not exist "$$FFTW_DLL_DST" md "$$FFTW_DLL_DST" $$escape_expand(\\n\\t))

	QMAKE_POST_LINK += $$quote(xcopy "$$FFTW_DLL_SRC\\*.dll" "$$FFTW_DLL_DST" /Y /I $$escape_expand(\\n\\t))

	cleanfftw.commands = for %%f in ("$$FFTW_DLL_DST\\libfftw*.dll") do del /Q "%%f"
	QMAKE_EXTRA_TARGETS += cleanfftw
	clean.depends += cleanfftw
}
