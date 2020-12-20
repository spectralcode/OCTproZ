#CUDA system and compiler settings

#path of cuda source files
SOURCEDIR = $$shell_path($$PWD/../src)
CUDA_SOURCES += $$SOURCEDIR/cuda_code.cu \

#cuda arch flags
#use sm_30 for max compatibility
#use sm_50 for Quadro K620, K620M, K1200, K2200, K2200M.
#use sm_61 for GeForce GTX 1060 ..1080 TI
CUDA_ARCH += sm_30 \
-gencode=arch=compute_30,code=sm_30 \
-gencode=arch=compute_50,code=sm_50 \
-gencode=arch=compute_52,code=sm_52 \
-gencode=arch=compute_60,code=sm_60 \
-gencode=arch=compute_61,code=sm_61 \
-gencode=arch=compute_70,code=sm_70 \

#nvcc compiler options
unix{
	NVCC_OPTIONS = --use_fast_math -std=c++11 --compiler-options -fPIC
}
win32{
	NVCC_OPTIONS = --use_fast_math
	#NVCC_OPTIONS = --use_fast_math -std=c++11 --compiler-options -fPIC
}

#cuda include paths
unix{
	CUDA_DIR = /usr/local/cuda

	INCLUDEPATH_CUDA += /usr/include/x86_64-linux-gnu/qt5 \	#todo: is there a more general way to access the qt include directory?
		/usr/include/x86_64-linux-gnu/qt5/QtCore \
		$$CUDA_DIR/include \
		$$CUDA_DIR/samples/common/inc

	#for aarch64 (jetson nano)
	INCLUDEPATH_CUDA += /usr/include/aarch64-linux-gnu/qt5 \
		/usr/include/aarch64-linux-gnu/qt5/QtCore

	INCLUDEPATH += $$CUDA_DIR/include \
		$$CUDA_DIR/samples/common/inc

	QMAKE_LIBDIR += $$CUDA_DIR/lib64
}
win32{
	CUDA_DIR = $$(CUDA_PATH)

	MSVCRT_LINK_FLAG_DEBUG  = "/MDd" # MSVCRT link option (MT: static, MD:dynamic. Must be the same as Qt SDK link option)
	MSVCRT_LINK_FLAG_RELEASE = "/MD"

	INCLUDEPATH += $$shell_path($$(NVCUDASAMPLES_ROOT)/common/inc) \ #this is needed for glew.h
		$$shell_path($$(NVCUDASAMPLES_ROOT)/common/lib/x64) \ #this is needed for OpenGL
		$$shell_path($$(CudaToolkitLibDir)) \
		$$CUDA_DIR/include \
		$$CUDA_DIR/common/inc \
		$$CUDA_DIR/../shared/inc

	INCLUDEPATH_CUDA += $$shell_path($$(NVCUDASAMPLES_ROOT)/common/inc) \ #this is needed for glew.h
		$$CUDA_DIR/include \
		$$CUDA_DIR/common/inc \
		$$CUDA_DIR/samples/common/inc

	# library directories
	SYSTEM_NAME = x64
	QMAKE_LIBDIR += $$CUDA_DIR/lib/$$SYSTEM_NAME \
		$$CUDA_DIR/common/lib/$$SYSTEM_NAME \
		$$CUDA_DIR/../shared/lib/$$SYSTEM_NAME
}
INCLUDEPATH_CUDA += $$shell_path($$(QTDIR)\include) \
	$$shell_path($$(QTDIR)\include\QtCore) \
	$$shell_path($(QTDIR)\lib)

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

#CUDA compiler configuration
SYSTEM_TYPE = 64
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
