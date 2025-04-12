DOCDIR = ../../../docs/site

#set path for documentation
unix{
	DOCEXPORTDIR = $$shell_path($$OUT_PWD/docs)
}
win32{
	CONFIG(debug, debug|release) {
		DOCEXPORTDIR = $$shell_path($$OUT_PWD/debug/docs)
	}
	CONFIG(release, debug|release) {
		DOCEXPORTDIR = $$shell_path($$OUT_PWD/release/docs)
	}
}

#create documentation folder
exists($$DOCEXPORTDIR){
	message("docdir already existing")
}else{
	unix{
		QMAKE_POST_LINK += $$quote(mkdir -p $${DOCEXPORTDIR} $$escape_expand(\\n\\t))
	}
	win32{
		QMAKE_POST_LINK += $$quote(md $${DOCEXPORTDIR} $$escape_expand(\\n\\t))
	}
}

#copy entire documentation folder recursively
unix{
	QMAKE_POST_LINK += $$quote(cp -R $$shell_path($$PWD/$$DOCDIR)/* $$DOCEXPORTDIR $$escape_expand(\\n\\t))
}
win32{
	#xcopy with
	#/E "copy all subdirectories, including empty ones",
	#/Y "suppresses prompts to confirm overwriting existing destination files",
	#/I "assume destination is a directory if copying more than one file"
	QMAKE_POST_LINK += $$quote(xcopy $$shell_path($$PWD/$$DOCDIR) $$DOCEXPORTDIR /E /Y /I $$escape_expand(\\n\\t))
}

#add docs folder to clean directive so that it will be deleted when running "make clean"
unix{
	cleandocs.commands = rm -rf $$shell_path($$DOCEXPORTDIR)
}
win32{
	cleandocs.commands = if exist $$shell_path($$DOCEXPORTDIR) rmdir /S /Q $$shell_path($$DOCEXPORTDIR)
}
clean.depends = cleandocs
QMAKE_EXTRA_TARGETS += clean cleandocs
