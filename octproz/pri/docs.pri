DOCDIR = ../../docs

DOC_FILES += \
	$$DOCDIR/index.html \

DOC_FILES_CSS += \
	$$DOCDIR/css/hyde.css \
	$$DOCDIR/css/poole.css \

DOC_FILES_IMAGES += \
	$$DOCDIR/images/math_linresampling.svg \
	$$DOCDIR/images/octproz_icon.png \
	$$DOCDIR/images/processing_bscanflip.png \
	$$DOCDIR/images/processing_pipeline.png \
	$$DOCDIR/images/processing_result.png \
	$$DOCDIR/images/processing_sinusoidalcorrection.png \
	$$DOCDIR/images/quickstart1.png \
	$$DOCDIR/images/quickstart2.png \
	$$DOCDIR/images/quickstart3.png \
	$$DOCDIR/images/quickstart4.png \
	$$DOCDIR/images/quickstart5.png \
	$$DOCDIR/images/quickstart6.png


unix{
	DOCEXPORTDIR = $$shell_path($$OUT_PWD/docs)
	DOCEXPORTDIR_CSS = $$shell_path($$OUT_PWD/docs/css)
	DOCEXPORTDIR_IMAGES = $$shell_path($$OUT_PWD/docs/images)
}
win32{
	CONFIG(debug, debug|release) {
		DOCEXPORTDIR = $$shell_path($$OUT_PWD/debug/docs)
		DOCEXPORTDIR_CSS = $$shell_path($$OUT_PWD/debug/docs/css)
		DOCEXPORTDIR_IMAGES = $$shell_path($$OUT_PWD/debug/docs/images)
	}
	CONFIG(release, debug|release) {
		DOCEXPORTDIR = $$shell_path($$OUT_PWD/release/docs)
		DOCEXPORTDIR_CSS = $$shell_path($$OUT_PWD/release/docs/css)
		DOCEXPORTDIR_IMAGES = $$shell_path($$OUT_PWD/release/docs/images)
	}
}


##Create DOCEXPORTDIR directory if not already existing
exists($$DOCEXPORTDIR){
	message("docdir already existing")
}else{
	QMAKE_POST_LINK += $$quote(mkdir -p $${DOCEXPORTDIR} $$escape_expand(\\n\\t))
	QMAKE_POST_LINK += $$quote(mkdir -p $${DOCEXPORTDIR_CSS} $$escape_expand(\\n\\t))
	QMAKE_POST_LINK += $$quote(mkdir -p $${DOCEXPORTDIR_IMAGES} $$escape_expand(\\n\\t))
}

##Copy documentaion folder to "DOCEXPORTDIR"
for(file, DOC_FILES):QMAKE_POST_LINK += $$QMAKE_COPY $$quote($$shell_path($$PWD/$${file})) $$quote($$DOCEXPORTDIR) $$escape_expand(\\n\\t)
for(file, DOC_FILES_CSS):QMAKE_POST_LINK += $$QMAKE_COPY $$quote($$shell_path($$PWD/$${file})) $$quote($$DOCEXPORTDIR_CSS) $$escape_expand(\\n\\t)
for(file, DOC_FILES_IMAGES):QMAKE_POST_LINK += $$QMAKE_COPY $$quote($$shell_path($$PWD/$${file})) $$quote($$DOCEXPORTDIR_IMAGES) $$escape_expand(\\n\\t)

##Add documentation files to clean directive. When running "make clean" documentaion will be deleted
#for(file, DOC_FILES):QMAKE_CLEAN += $$shell_path($$DOCEXPORTDIR/$${file}) #todo: this does not work probably because DOC_FILES contains the full paths of the files and not just the file name. Find a nice and easy way to add doc files to QMAKE_CLEAN
