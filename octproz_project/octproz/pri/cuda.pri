#CUDA system and compiler settings

#path of cuda source files
SOURCEDIR = $$shell_path($$PWD/../src)
CUDA_SOURCES += $$SOURCEDIR/cuda_code.cu \

#cuda architecture flags
#change these flags according to your GPU
#see: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

##use this for maximum compatibility with CUDA 11.0
#CUDA_ARCH += sm_52 \
#-gencode=arch=compute_52,code=sm_52 \
#-gencode=arch=compute_60,code=sm_60 \
#-gencode=arch=compute_61,code=sm_61 \
#-gencode=arch=compute_70,code=sm_70 \
#-gencode=arch=compute_75,code=sm_75 \
#-gencode=arch=compute_80,code=sm_80 \
#-gencode=arch=compute_86,code=sm_86 \
#-gencode=arch=compute_86,code=compute_86

##use this for maximum compatibility with CUDA 12.8
CUDA_ARCH += sm_52 \
-gencode=arch=compute_52,code=sm_52 \
-gencode=arch=compute_60,code=sm_60 \
-gencode=arch=compute_61,code=sm_61 \
-gencode=arch=compute_70,code=sm_70 \
-gencode=arch=compute_75,code=sm_75 \
-gencode=arch=compute_80,code=sm_80 \
-gencode=arch=compute_86,code=sm_86 \
-gencode=arch=compute_87,code=sm_87 \
-gencode=arch=compute_89,code=sm_89 \
-gencode=arch=compute_90,code=sm_90 \
-gencode=arch=compute_100,code=sm_100 \
-gencode=arch=compute_120,code=sm_120 \
-gencode=arch=compute_120,code=compute_120

#if the host architecture is aarch64, assume Jetson Nano and override CUDA_ARCH accordingly
message(host architecture is: $$QMAKE_HOST.arch)
contains(QMAKE_HOST.arch, aarch64){
	CUDA_ARCH = sm_53 \
	-gencode=arch=compute_53,code=sm_53 \
	-gencode=arch=compute_53,code=compute_53
}



#include addtional macro definitions
include(../../config.pri)
CUDA_DEFINES_FLAGS = $$join(CUDA_RELEVANT_DEFINES, '-D', '-D', '')

#nvcc compiler options
unix{
	NVCC_OPTIONS = --use_fast_math -std=c++11 --compiler-options -fPIC
}
win32{
	NVCC_OPTIONS = --use_fast_math -diag-suppress 1723,1394 #diag-suppress is used to ignore Qt-related warnings coming from Qt's header files
}

#cuda include paths
unix{
	CUDA_DIR = /usr/local/cuda
	QMAKE_LIBDIR += $$CUDA_DIR/lib64
	exists($$shell_path($$CUDA_DIR/samples)){
		NVCUDASAMPLES_ROOT = $$shell_path($$CUDA_DIR/samples)
	} else {
		NVCUDASAMPLES_ROOT = $$shell_path($$PWD/../../thirdparty/cuda)
	}
}
win32{
	CUDA_DIR = $$(CUDA_PATH)

	MSVCRT_LINK_FLAG_DEBUG  = "/MDd" # MSVCRT link option (MT: static, MD:dynamic. Must be the same as Qt SDK link option)
	MSVCRT_LINK_FLAG_RELEASE = "/MD"

	isEmpty(NVCUDASAMPLES_ROOT){
		NVCUDASAMPLES_ROOT = $$shell_path($$PWD/../../thirdparty/cuda) #in older CUDA versions NVCUDASAMPLES_ROOT is defined. it is the location of the CUDA samples folder
	}

	INCLUDEPATH += $$shell_path($$(NVCUDASAMPLES_ROOT)/common/inc)

	INCLUDEPATH_CUDA += $$shell_path($$(NVCUDASAMPLES_ROOT)/common/inc)

	#library directories
	SYSTEM_NAME = x64
	QMAKE_LIBDIR += $$CUDA_DIR/lib/$$SYSTEM_NAME
}
INCLUDEPATH += $$CUDA_DIR/include \
	$$NVCUDASAMPLES_ROOT/common/inc

INCLUDEPATH_CUDA += $$[QT_INSTALL_HEADERS] \
	$$[QT_INSTALL_HEADERS]/QtCore \
	$$CUDA_DIR/include \
	$$NVCUDASAMPLES_ROOT/common/inc

#cuda libraries
unix{
	CUDA_LIBS += -lcudart -lcuda -lcufft -lculibos
}
win32{
	CUDA_LIBS += -lcudart -lcuda -lcufft
}
LIBS += $$CUDA_LIBS

#put quotation marks to path names (to avoid problems with spaces in path names)
CUDA_INC = $$join(INCLUDEPATH_CUDA,'" -I"','-I"','"')

#compiler flags
unix{
	QMAKE_LFLAGS += -Wl,-rpath,$$CUDA_DIR/lib
	NVCCFLAGS = -Xlinker -rpath,$$CUDA_DIR/lib
	#NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v
}

#cuda compiler configuration
SYSTEM_TYPE = 64
CUDA_OBJECTS_DIR = ./
unix{
	CONFIG(debug, debug|release) {
		cuda_d.input = CUDA_SOURCES
		cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
		cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_DEFINES_FLAGS $$CUDA_INC --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
		cuda_d.dependency_type = TYPE_C
		QMAKE_EXTRA_COMPILERS += cuda_d
	}
	CONFIG(release, debug|release) {
		cuda.input = CUDA_SOURCES
		cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
		cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_DEFINES_FLAGS $$CUDA_INC --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
		cuda.dependency_type = TYPE_C
		QMAKE_EXTRA_COMPILERS += cuda
	}
}
win32{
	CONFIG(debug, debug|release) {
		cuda_d.input = CUDA_SOURCES
		cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
		cuda_d.commands = $$CUDA_DIR/bin/nvcc.exe -D_DEBUG $$NVCC_OPTIONS $$CUDA_DEFINES_FLAGS $$CUDA_INC --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
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
		cuda.commands = $$CUDA_DIR/bin/nvcc.exe $$NVCC_OPTIONS $$CUDA_DEFINES_FLAGS $$CUDA_INC --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
					--compile -cudart static -DWIN32 -D_MBCS \
					-Xcompiler "/wd4819,/EHsc,/W3,/nologo,/O2,/Zi" \
					-Xcompiler $$MSVCRT_LINK_FLAG_RELEASE \
					-c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
		cuda.dependency_type = TYPE_C
		QMAKE_EXTRA_COMPILERS += cuda
	}
}

message(CUDA_INC is $$CUDA_INC)
