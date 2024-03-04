#-------------------------------------------------
#
# Project created by QtCreator 2018-02-15T16:57:08
#
#-------------------------------------------------

QT += core gui widgets

TARGET = OCTproZ_DevKit
TEMPLATE = lib
CONFIG += staticlib

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000	# disables all the APIs deprecated before Qt 6.0.0


#define path of OCTproZ_DevKit share directory
SHAREDIR = $$shell_path($$PWD/../octproz_share_dev)

SOURCES += \
	src/octproz_devkit.cpp \
	src/acquisitionbuffer.cpp \
	src/acquisitionparameter.cpp \
	src/acquisitionsystem.cpp \
	src/extension.cpp


HEADERS += \
	src/octproz_devkit.h \
	src/acquisitionbuffer.h \
	src/acquisitionparameter.h \
	src/acquisitionsystem.h \
	src/extension.h \
	src/plugin.h

unix {
	target.path = /usr/lib
	INSTALLS += target

}

CONFIG(debug, debug|release) {
	SHAREDIR_LIB = $$shell_path($$SHAREDIR/debug)
	unix{
		OUTFILE = $$shell_path($$OUT_PWD/lib$$TARGET'.'$${QMAKE_EXTENSION_STATICLIB})
	}
	win32{
		OUTFILE = $$shell_path($$OUT_PWD/debug/$$TARGET'.'$${QMAKE_EXTENSION_STATICLIB})
	}
}
CONFIG(release, debug|release) {
	SHAREDIR_LIB = $$shell_path($$SHAREDIR/release)
	unix{
		OUTFILE = $$shell_path($$OUT_PWD/lib$$TARGET'.'$${QMAKE_EXTENSION_STATICLIB})
	}
	win32{
		OUTFILE = $$shell_path($$OUT_PWD/release/$$TARGET'.'$${QMAKE_EXTENSION_STATICLIB})
	}
}


message(sharedir is $$SHAREDIR)
message(outfile is $$OUTFILE)


##Create SHAREDIR directory if not already existing
exists($$SHAREDIR){
	message("sharedir already existing")
}else{
	QMAKE_PRE_LINK += $$quote(mkdir $${SHAREDIR} $$escape_expand(\\n\\t))
}
exists($$SHAREDIR_LIB){
	message("sharedir_lib already existing")
}else{
	QMAKE_PRE_LINK += $$quote(mkdir $${SHAREDIR_LIB} $$escape_expand(\\n\\t))
}

##Copy headerfiles and static lib to "SHAREDIR"
for(header, HEADERS):QMAKE_POST_LINK += $$QMAKE_COPY $$quote($$shell_path($$PWD/$${header})) $$quote($$SHAREDIR) $$escape_expand(\\n\\t)
QMAKE_POST_LINK += $$QMAKE_COPY $$quote($${OUTFILE}) $$quote($$SHAREDIR_LIB) $$escape_expand(\\n\\t)


##Add SHAREDIR to clean directive. When running "make clean" SHAREDIR will then also be deleted
unix {
	#QMAKE_CLEAN += -r $$SHAREDIR
	for(header, HEADERS):QMAKE_CLEAN += $$shell_path($$SHAREDIR/$${header})
	QMAKE_CLEAN += $$shell_path($$SHAREDIR_LIB/lib$$TARGET'.'$${QMAKE_EXTENSION_STATICLIB})
}
win32 {
	for(header, HEADERS):QMAKE_CLEAN += $$shell_path($$SHAREDIR/$${header})
	QMAKE_CLEAN += $$shell_path($$SHAREDIR_LIB/$$TARGET'.'$${QMAKE_EXTENSION_STATICLIB})
}

#include cuda.pri only for Jetson Nano
message("host architecture is: " $$QMAKE_HOST.arch)
contains(QMAKE_HOST.arch, aarch64){
	include(../octproz/pri/cuda.pri)
}
