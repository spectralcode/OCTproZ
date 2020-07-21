QT 	  += core gui widgets printsupport

TARGET = OCTproZ
TEMPLATE = app


#define path of OCTproZ_DevKit share directory
SHAREDIR = $$shell_path($$PWD/../octproz_share_dev)
QCUSTOMPLOTDIR = $$shell_path($$PWD/../QCustomPlot)
SOURCEDIR = $$shell_path($$PWD/src)


INCLUDEPATH +=  \
	$$SOURCEDIR \
	$$SHAREDIR \
	$$QCUSTOMPLOTDIR

SOURCES += \
	$$SOURCEDIR/aboutdialog.cpp \
	$$SOURCEDIR/glwindow3d.cpp \
	$$SOURCEDIR/main.cpp \
	$$SOURCEDIR/mesh.cpp \
	$$SOURCEDIR/minicurveplot.cpp \
	$$SOURCEDIR/octproz.cpp \
	$$SOURCEDIR/raycastvolume.cpp \
	$$SOURCEDIR/systemmanager.cpp \
	$$QCUSTOMPLOTDIR/qcustomplot.cpp \
	$$SOURCEDIR/systemchooser.cpp \
	$$SOURCEDIR/messageconsole.cpp \
	$$SOURCEDIR/glwindow2d.cpp \
	$$SOURCEDIR/plotwindow1d.cpp \
	$$SOURCEDIR/processing.cpp \
	$$SOURCEDIR/sidebar.cpp \
	$$SOURCEDIR/octalgorithmparameters.cpp \
	$$SOURCEDIR/settings.cpp \
	$$SOURCEDIR/polynomial.cpp \
	$$SOURCEDIR/extensionmanager.cpp \
	$$SOURCEDIR/trackball.cpp \
	$$SOURCEDIR/windowfunction.cpp \
	$$SOURCEDIR/gpu2hostnotifier.cpp \
	$$SOURCEDIR/eventguard.cpp \
	$$SOURCEDIR/recorder.cpp \
	$$SOURCEDIR/stringspinbox.cpp \
	$$SOURCEDIR/controlpanel.cpp \
	$$SOURCEDIR/extensioneventfilter.cpp

	unix{
		SOURCES += $$SOURCEDIR/cuda_code.cu
		SOURCES -= $$SOURCEDIR/cuda_code.cu
	}

HEADERS += \
	$$SOURCEDIR/aboutdialog.h \
	$$SOURCEDIR/glwindow3d.h \
	$$SOURCEDIR/mesh.h \
	$$SOURCEDIR/minicurveplot.h \
	$$SOURCEDIR/octproz.h \
	$$SHAREDIR/octproz_devkit.h \
	$$SHAREDIR/acquisitionbuffer.h \
	$$SHAREDIR/acquisitionsystem.h \
	$$SOURCEDIR/raycastvolume.h \
	$$SOURCEDIR/systemmanager.h \
	$$QCUSTOMPLOTDIR/qcustomplot.h \
	$$SOURCEDIR/kernels.h \
	$$SOURCEDIR/systemchooser.h \
	$$SOURCEDIR/messageconsole.h \
	$$SOURCEDIR/glwindow2d.h \
	$$SOURCEDIR/plotwindow1d.h \
	$$SOURCEDIR/processing.h \
	$$SOURCEDIR/sidebar.h \
	$$SOURCEDIR/octalgorithmparameters.h \
	$$SOURCEDIR/settings.h \
	$$SOURCEDIR/polynomial.h \
	$$SOURCEDIR/extensionmanager.h \
	$$SOURCEDIR/trackball.h \
	$$SOURCEDIR/windowfunction.h \
	$$SOURCEDIR/gpu2hostnotifier.h \
	$$SOURCEDIR/eventguard.h \
	$$SOURCEDIR/recorder.h \
	$$SOURCEDIR/stringspinbox.h \
	$$SOURCEDIR/controlpanel.h \
	$$SOURCEDIR/extensioneventfilter.h

FORMS += \
	$$SOURCEDIR/octproz.ui \
	$$SOURCEDIR/sidebar.ui

RESOURCES += \
	resources.qrc

DISTFILES += \
	$$SOURCEDIR/cuda_code.cu \

#OCTproZ_DevKit libraries
CONFIG(debug, debug|release) {
	unix{
		LIBS += $$shell_path($$SHAREDIR/debug/libOCTproZ_DevKit.a)
	}
	win32{
		LIBS += $$shell_path($$SHAREDIR/debug/OCTproZ_DevKit.lib) 
	}
}
CONFIG(release, debug|release) {
	unix{
		LIBS += $$shell_path($$SHAREDIR/release/libOCTproZ_DevKit.a)
	}
	win32{
		LIBS += $$shell_path($$SHAREDIR/release/OCTproZ_DevKit.lib)
	}
}

#Open GL libs
unix{
	LIBS += -lGL -lGLU -lX11 -lglut
}
win32{
	LIBS += -lopengl32 -lglu32
}


#C++ flags
unix{
	#QMAKE_CXXFLAGS_RELEASE =-O3
}

#include cuda configuration
include(pri/cuda.pri)

#include pri file to copy documentation to build folder
include(pri/docs.pri)

#set application icon
RC_ICONS = icons/OCTproZ_icon.ico
