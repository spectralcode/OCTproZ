QT 	  += core gui opengl widgets printsupport

TARGET = OCTproZ
TEMPLATE = app


#define path of OCTproZ_DevKit share directory
SHAREDIR = $$shell_path($$PWD/../octproz_share_dev)
QCUSTOMPLOTDIR = $$shell_path($$PWD/../QCustomPlot)
SOURCEDIR = $$shell_path($$PWD/src)


win32{
	# MSVCRT link option (MT: static, MD:dynamic. Must be the same as Qt SDK link option)
	MSVCRT_LINK_FLAG_DEBUG   = "/MDd"
	MSVCRT_LINK_FLAG_RELEASE = "/MD"
	INCLUDEPATH += $$shell_path($$(NVCUDASAMPLES_ROOT)/common/inc) #this is needed for glew.h
	INCLUDEPATH += $$shell_path($$(NVCUDASAMPLES_ROOT)/common/lib/x64) #this is needed for OpenGL
	INCLUDEPATH += $$shell_path($$(CudaToolkitLibDir))
	INCLUDEPATH_CUDA += $$shell_path($$(NVCUDASAMPLES_ROOT)/common/inc) #this is needed for glew.h
}
	INCLUDEPATH_CUDA += $$shell_path($$(QTDIR)\include)
	INCLUDEPATH_CUDA += $$shell_path($$(QTDIR)\include\QtCore)
	INCLUDEPATH_CUDA += $$shell_path($(QTDIR)\lib)

message(qtdir is $$QTDIR)


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

#OCTproZ_DevKit libraries
CONFIG(debug, debug|release) {
	unix{
		LIBS += $$shell_path($$SHAREDIR/debug/libOCTproZ_DevKit.a)
	}
	win32{
		LIBS += $$shell_path($$SHAREDIR/debug/OCTproZ_DevKit.lib) 
	}
	#NVCC_LIBS += -lQt5Cored
}
CONFIG(release, debug|release) {
	unix{
		LIBS += $$shell_path($$SHAREDIR/release/libOCTproZ_DevKit.a)
	}
	win32{
		LIBS += $$shell_path($$SHAREDIR/release/OCTproZ_DevKit.lib)
	}
	#NVCC_LIBS += -lQt5Core
}



#Open GL libs
unix{
	LIBS += -lGL -lGLU -lX11 -lglut
}
win32{
	LIBS += -lopengl32 -lglu32
}


DISTFILES += \
	$$SOURCEDIR/cuda_code.cu \

CUDA_SOURCES += $$SOURCEDIR/cuda_code.cu \



#C++ flags
unix{
	#QMAKE_CXXFLAGS_RELEASE =-O3
}

#Path to cuda toolkit install
unix{
	CUDA_DIR = /usr/local/cuda
}
win32{
	CUDA_DIR = $$(CUDA_PATH)
}


#CUDA system/compiler settings
SYSTEM_TYPE = 64
#CUDA_ARCH = sm_61 		  #Type of CUDA architecture, use sm_61 for GeForce GTX 1060 ..1080 TI
#CUDA_ARCH = sm_50 		 #use sm_50 for Quadro K620, K620M, K1200, K2200, K2200M

########
#cuda 8 arch flags for maximum compatibility (probably not maximum efficiency)
CUDA_ARCH += sm_30 \
-gencode=arch=compute_30,code=sm_30 \
-gencode=arch=compute_50,code=sm_50 \
-gencode=arch=compute_52,code=sm_52 \
-gencode=arch=compute_60,code=sm_60 \
-gencode=arch=compute_61,code=sm_61 \
-gencode=arch=compute_70,code=sm_70 \
########

unix{
	NVCC_OPTIONS = --use_fast_math -std=c++11 --compiler-options -fPIC
}
win32{
	NVCC_OPTIONS = --use_fast_math
	#NVCC_OPTIONS = --use_fast_math -std=c++11 --compiler-options -fPIC
}



# Path to CUDA header and libs files
unix{
	INCLUDEPATH_CUDA  += $$CUDA_DIR/include \
						 $$CUDA_DIR/samples/common/inc
	INCLUDEPATH  += $$CUDA_DIR/include
	INCLUDEPATH  += $$CUDA_DIR/samples/common/inc
	QMAKE_LIBDIR += $$CUDA_DIR/lib64
}
win32{
	INCLUDEPATH_CUDA  += $$CUDA_DIR/include \
						 $$CUDA_DIR/common/inc \
						 $$CUDA_DIR/samples/common/inc
	INCLUDEPATH += $$CUDA_DIR/include \
				   $$CUDA_DIR/common/inc \
				   $$CUDA_DIR/../shared/inc

	# library directories
	SYSTEM_NAME = x64
	QMAKE_LIBDIR += $$CUDA_DIR/lib/$$SYSTEM_NAME \
					$$CUDA_DIR/common/lib/$$SYSTEM_NAME \
					$$CUDA_DIR/../shared/lib/$$SYSTEM_NAME
}


#cuda libraries
unix{
	CUDA_LIBS += -lcudart -lcuda -lcufft -lculibos
}
win32{
	CUDA_LIBS += -lcudart -lcuda -lcufft
}


# Put quotation marks to path names (to avoid problems with spaces in path names)
CUDA_INC = $$join(INCLUDEPATH_CUDA,'" -I"','-I"','"')
#LIBS += -lATSApi# AlazarTech lib for digitizer
LIBS += $$CUDA_LIBS




unix{
	# SPECIFY THE R PATH FOR NVCC
	QMAKE_LFLAGS += -Wl,-rpath,$$CUDA_DIR/lib
	NVCCFLAGS = -Xlinker -rpath,$$CUDA_DIR/lib
	#NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v
}


#CUDA compiler configuration
CUDA_OBJECTS_DIR = ./
unix{
	CONFIG(debug, debug|release) {
		cuda_d.input = CUDA_SOURCES
		cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
		cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
		cuda_d.dependency_type = TYPE_C
		QMAKE_EXTRA_COMPILERS += cuda_d
	}
	CONFIG(release, debug|release) {
		cuda.input = CUDA_SOURCES
		cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
		cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
		cuda.dependency_type = TYPE_C
		QMAKE_EXTRA_COMPILERS += cuda
	}
}
win32{
	CONFIG(debug, debug|release) {
		cuda_d.input = CUDA_SOURCES
		cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
		cuda_d.commands = $$CUDA_DIR/bin/nvcc.exe -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
					  --compile -cudart static -g -DWIN32 -D_MBCS \
					  -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/Od,/Zi,/RTC1" \
					  -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG \
					  -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}

		cuda_d.dependency_type = TYPE_C
		QMAKE_EXTRA_COMPILERS += cuda_d
	}
	CONFIG(release, debug|release) {
		cuda.input = CUDA_SOURCES
		cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
		cuda.commands = $$CUDA_DIR/bin/nvcc.exe $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
					--compile -cudart static -DWIN32 -D_MBCS \
					-Xcompiler "/wd4819,/EHsc,/W3,/nologo,/O2,/Zi" \
					-Xcompiler $$MSVCRT_LINK_FLAG_RELEASE \
					-c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
		cuda.dependency_type = TYPE_C
		QMAKE_EXTRA_COMPILERS += cuda
	}
}


message(NVCClibs are $$NVCC_LIBS)

RC_ICONS = icons/OCTproZ_icon.ico
