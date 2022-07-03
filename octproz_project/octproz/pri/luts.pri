LUTDIR = ../luts

LUT_FILES += \
	$$LUTDIR/blue_lut.png \
	$$LUTDIR/fire_lut.png \
	$$LUTDIR/hotter_lut.png \
	$$LUTDIR/ice_lut.png \
	$$LUTDIR/six_shades_lut.png \
	$$LUTDIR/sixteen_colors_lut.png \
	$$LUTDIR/deep_red_lut.png \
	$$LUTDIR/deep_blue_lut.png

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

##Create lut directory
exists($$LUTEXPORTDIR){
	message("lutdir already existing")
}else{
	unix{
		QMAKE_POST_LINK += $$quote(mkdir -p $${LUTEXPORTDIR} $$escape_expand(\\n\\t))
	}
	win32{
		QMAKE_POST_LINK += $$quote(md $${LUTEXPORTDIR} $$escape_expand(\\n\\t))
	}
}

##Copy lookup table folder to "LUTEXPORTDIR"
for(file, LUT_FILES):QMAKE_POST_LINK += $$QMAKE_COPY $$quote($$shell_path($$PWD/$${file})) $$quote($$LUTEXPORTDIR) $$escape_expand(\\n\\t)

##Add lookup table files to clean directive. When running "make clean" lookup table files will be deleted
#for(file, LUT_FILES):QMAKE_CLEAN += $$shell_path($$LUTEXPORTDIR/$${file}) #todo: this does not work probably because LUT_FILES contains the full paths of the files and not just the file name. Find a nice and easy way to add doc files to QMAKE_CLEAN
