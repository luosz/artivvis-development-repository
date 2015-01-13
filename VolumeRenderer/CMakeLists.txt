cmake_minimum_required(VERSION 2.8.11)

project("VolumeRenderer")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

## Cache variables (user input may be needed)
set(VOLUME_RENDERER_SRC "VolumeRenderer/" CACHE STRING "Source code directory." FORCE)
set(TRANSFER_FUNCTION_EDITOR_SRC "TransferFunctionEditor/" CACHE STRING "Source code directory." FORCE)
set(GLM_SRC "gl/glm/" CACHE STRING "Source headers for the GLM." FORCE)
set(INCLUDE_DIR "gl/include" CACHE STRING "Source headers for the additional includes." FORCE)
set(QT_INSTALL_PREFIX "THIS MUST BE SET" CACHE PATH "Install directory for Qt")

## Local variables
set(CMAKE_PREFIX_PATH ${QT_INSTALL_PREFIX})
set(CUDA_LIB ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so)

## Required libraries (Search handled by CMake)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(CUDA REQUIRED)
find_package(DevIL REQUIRED)

## Additional include directories
include_directories(${GLM_SRC})
include_directories(${IL_INCLUDE_DIR})
include_directories(${INCLUDE_DIR})
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include/)
include_directories(${VOLUME_RENDERER_SRC}GeneratedFiles/)
link_directories(${CUDA_LIB_DIR})

## Addind definition to fail on first error. Makes it easier to read.
add_definitions( -Wfatal-errors )

## Adding definition to suppress GLM internal warning.
add_definitions( -DGLM_FORCE_RADIANS )

## Create the TransferFunctionEditor as a library
file(GLOB TF_EDITOR_SOURCES
	"${TRANSFER_FUNCTION_EDITOR_SRC}*.cpp"
	"${TRANSFER_FUNCTION_EDITOR_SRC}*.h"
)
add_library(TransferFunctionEditor ${TF_EDITOR_SOURCES})
target_link_libraries(
	TransferFunctionEditor
	Qt5::Core 
	Qt5::Widgets
	${GLUT_LIBRARIES} 
	${OPENGL_LIBRARIES}
	${GLEW_LIBRARIES}
	${IL_LIBRARIES}
	${CUDA_OBJECTS}
)

## Find standard C objects to build
file(GLOB SOURCES 
	"${INCLUDE_DIR}/tinyxml2.cpp"
	"${VOLUME_RENDERER_SRC}*.cu"
	"${VOLUME_RENDERER_SRC}*.cpp"
	"${VOLUME_RENDERER_SRC}*.h"
)
cuda_include_directories(${Qt5Widgets_INCLUDE_DIRS})

## Build CUDA Objects
set(CUDA_NVCC_FLAGS "--compiler-options '-fPIC'" CACHE STRING "nvcc flags" FORCE)

## Build everything
cuda_add_executable(VolumeRenderer ${SOURCES})

target_link_libraries(
	VolumeRenderer 
	TransferFunctionEditor
	Qt5::Core 
	Qt5::Widgets
	${CUDA_CUDART_LIBRARY}
	${GLUT_LIBRARIES} 
	${OPENGL_LIBRARIES}
	${GLEW_LIBRARIES}
	${IL_LIBRARIES}
)

## Copy shader files to executable directory
file(GLOB SHADER_FILES "${VOLUME_RENDERER_SRC}*.txt")
file(COPY ${SHADER_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
