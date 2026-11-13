#HIP system and compiler settings for AMD GPUs
#Created by Claude for the ROCm port of OCTproZ

#path of HIP source files
SOURCEDIR = $$shell_path($$PWD/../src)
HIP_SOURCES += $$SOURCEDIR/cuda_code.cu

#HIP architecture flags
#Set via environment HIP_ARCHITECTURES or default to gfx90a
isEmpty(HIP_ARCHITECTURES){
    HIP_ARCH = gfx90a
} else {
    HIP_ARCH = $$HIP_ARCHITECTURES
}

#if HIP_ARCH is not set via environment, check for command-line override
!isEmpty(HIP_ARCH_OVERRIDE){
    HIP_ARCH = $$HIP_ARCH_OVERRIDE
}

message(Building for HIP architecture: $$HIP_ARCH)

#HIP directory (ROCm installation path)
unix{
    HIP_DIR = /opt/rocm
    QMAKE_LIBDIR += $$HIP_DIR/lib
}
win32{
    HIP_DIR = $$(HIP_PATH)
    isEmpty(HIP_DIR){
        HIP_DIR = C:/AMD/ROCm/6.3
    }
    QMAKE_LIBDIR += $$HIP_DIR/lib
}

#HIP include paths
INCLUDEPATH += $$HIP_DIR/include \
    $$HIP_DIR/include/hip

INCLUDEPATH_HIP += $$[QT_INSTALL_HEADERS] \
    $$[QT_INSTALL_HEADERS]/QtCore \
    $$HIP_DIR/include \
    $$HIP_DIR/include/hip

#HIP libraries
unix{
    HIP_LIBS += -lamdhip64 -lhipfft
}
win32{
    HIP_LIBS += -lamdhip64 -lhipfft
}
LIBS += $$HIP_LIBS

#Define USE_HIP and __HIP_PLATFORM_AMD__ for the compat header
#__HIP_PLATFORM_AMD__ is needed so hip headers select the AMD code path
DEFINES += USE_HIP __HIP_PLATFORM_AMD__

#include additional macro definitions
include(../../config.pri)
HIP_DEFINES_FLAGS = $$join(CUDA_RELEVANT_DEFINES, ' -D', '-D', '')

#put quotation marks to path names (to avoid problems with spaces in path names)
HIP_INC = $$join(INCLUDEPATH_HIP,'" -I"','-I"','"')

#hipcc compiler options
unix{
    HIPCC_OPTIONS = -ffast-math -std=c++17 -fPIC
}
win32{
    HIPCC_OPTIONS = -ffast-math -std=c++17
}

#compiler flags
unix{
    QMAKE_LFLAGS += -Wl,-rpath,$$HIP_DIR/lib
}

#HIP compiler configuration
SYSTEM_TYPE = 64
HIP_OBJECTS_DIR = ./

unix{
    CONFIG(debug, debug|release) {
        hip_d.input = HIP_SOURCES
        hip_d.output = $$HIP_OBJECTS_DIR/${QMAKE_FILE_BASE}_hip.o
        hip_d.commands = $$HIP_DIR/bin/hipcc -D_DEBUG -DUSE_HIP $$HIPCC_OPTIONS $$HIP_DEFINES_FLAGS $$HIP_INC --offload-arch=$$HIP_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
        hip_d.dependency_type = TYPE_C
        QMAKE_EXTRA_COMPILERS += hip_d
    }
    CONFIG(release, debug|release) {
        hip.input = HIP_SOURCES
        hip.output = $$HIP_OBJECTS_DIR/${QMAKE_FILE_BASE}_hip.o
        hip.commands = $$HIP_DIR/bin/hipcc -DUSE_HIP $$HIPCC_OPTIONS $$HIP_DEFINES_FLAGS $$HIP_INC --offload-arch=$$HIP_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
        hip.dependency_type = TYPE_C
        QMAKE_EXTRA_COMPILERS += hip
    }
}

win32{
    CONFIG(debug, debug|release) {
        hip_d.input = HIP_SOURCES
        hip_d.output = $$HIP_OBJECTS_DIR/${QMAKE_FILE_BASE}_hip.obj
        hip_d.commands = $$HIP_DIR/bin/hipcc.exe -D_DEBUG -DUSE_HIP $$HIPCC_OPTIONS $$HIP_DEFINES_FLAGS $$HIP_INC --offload-arch=$$HIP_ARCH \
                    -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
        hip_d.dependency_type = TYPE_C
        QMAKE_EXTRA_COMPILERS += hip_d
    }
    CONFIG(release, debug|release) {
        hip.input = HIP_SOURCES
        hip.output = $$HIP_OBJECTS_DIR/${QMAKE_FILE_BASE}_hip.obj
        hip.commands = $$HIP_DIR/bin/hipcc.exe -DUSE_HIP $$HIPCC_OPTIONS $$HIP_DEFINES_FLAGS $$HIP_INC --offload-arch=$$HIP_ARCH \
                    -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
        hip.dependency_type = TYPE_C
        QMAKE_EXTRA_COMPILERS += hip
    }
}

message(HIP_INC is $$HIP_INC)
