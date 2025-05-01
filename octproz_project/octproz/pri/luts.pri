LUTDIR = ../luts

#set path for lookup tables
unix{
	LUTEXPORTDIR = $$shell_path($$OUT_PWD/luts)
}
win32{
	CONFIG(debug, debug|release) {
		LUTEXPORTDIR = $$shell_path($$OUT_PWD/debug/luts)
	}
	CONFIG(release, debug|release) {
		LUTEXPORTDIR = $$shell_path($$OUT_PWD/release/luts)
	}
}

#create lookup tables folder
exists($$LUTEXPORTDIR){
	message("lutdir already existing")
}else{
	unix{
		QMAKE_POST_LINK += $$quote(mkdir -p $${LUTEXPORTDIR} $$escape_expand(\\n\\t))
	}
	win32{
		QMAKE_POST_LINK += $$quote(if not exist "$${LUTEXPORTDIR}\\." md "$${LUTEXPORTDIR}" $$escape_expand(\\n\\t))
	}
}

#copy all lookup table files from source directory to destination folder
unix{
	QMAKE_POST_LINK += $$quote(cp -R $$shell_path($$PWD/$$LUTDIR)/* $$LUTEXPORTDIR $$escape_expand(\\n\\t))
}
win32{
	#xcopy with
	#/E "copy all subdirectories, including empty ones",
	#/Y "suppresses prompts to confirm overwriting existing destination files",
	#/I "assume destination is a directory if copying more than one file"
	QMAKE_POST_LINK += $$quote(xcopy $$shell_path($$PWD/$$LUTDIR) $$LUTEXPORTDIR /E /Y /I $$escape_expand(\\n\\t))
}

#add luts folder to clean directive so that it will be deleted when running "make clean"
unix{
	cleanluts.commands = rm -rf $$shell_path($$LUTEXPORTDIR)
}
win32{
	cleanluts.commands = if exist $$shell_path($$LUTEXPORTDIR) rmdir /S /Q $$shell_path($$LUTEXPORTDIR)
}
clean.depends += cleanluts
QMAKE_EXTRA_TARGETS += clean cleanluts
