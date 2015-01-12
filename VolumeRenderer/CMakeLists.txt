cmake_minimum_required(VERSION 2.8.11)

project("VolumeRenderer")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

## Cache variables (user input may be needed)
set(VOLUME_RENDERER_SRC "VolumeRenderer/" CACHE STRING "Source code directory." FORCE)
set(TRANSFER_FUNCTION_EDITOR_SRC "TransferFunctionEditor/" CACHE STRING "Source code directory." FORCE)
set(GLM_SRC "gl/glm/" CACHE STRING "Source headers for the GLM." FORCE)
set(DEVIL_SRC "gl/include/IL/" CACHE STRING "Source headers for the DevIL library.")
set(INCLUDE_DIR "gl/include" CACHE STRING "Source headers for the additional includes." FORCE)

## Local variables
set(CMAKE_PREFIX_PATH ${QT_INSTALL_PREFIX})
set(CUDA_LIB ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so)

## Required libraries
find_package(GLUT)
find_package(GLEW)
find_package(OpenGL)
find_package(Qt5Widgets)
find_package(Qt5Core)
find_package(CUDA)
find_package(DevIL)

## Additional include directories
include_directories(${GLM_SRC})
include_directories(${DEVIL_SRC})
include_directories(${INCLUDE_DIR})
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include/)
include_directories(${VOLUME_RENDERER_SRC}GeneratedFiles/)
link_directories(${CUDA_LIB_DIR})

## Addind definition to fail on first error. Makes it easier to read.
add_definitions( -Wfatal-errors )

## Adding definition to suppress GLM internal warning.
add_definitions( -DGLM_FORCE_RADIANS )

## Find standard C objects to build
FILE(GLOB SOURCES 
	"${INCLUDE_DIR}/tinyxml2.cpp"
	"${VOLUME_RENDERER_SRC}*.cu"
	"${VOLUME_RENDERER_SRC}*.cpp"
	"${VOLUME_RENDERER_SRC}*.h"
	"${TRANSFER_FUNCTION_EDITOR_SRC}*.cpp"
	"${TRANSFER_FUNCTION_EDITOR_SRC}*.h"
)
add_executable(VolumeRenderer ${SOURCES})

## Build CUDA Objects
FILE(GLOB CU_SOURCES 
	"${VOLUME_RENDERER_SRC}*.cu"
)
SET(CUDA_NVCC_FLAGS "--compiler-options '-fPIC'" CACHE STRING "nvcc flags" FORCE)
cuda_include_directories(${Qt5Widgets_INCLUDE_DIRS})
cuda_compile(CUDA_OBJECTS ${CU_SOURCES})

## Build everything
target_link_libraries(
	VolumeRenderer 
	Qt5::Core 
	Qt5::Widgets 
	${CUDA_LIB} 
	${GLUT_LIBRARIES} 
	${OPENGL_LIBRARIES}
	${GLEW_LIBRARIES}
	${IL_LIBRARIES}
	${CUDA_OBJECTS}
)

## Copy shader files to executable directory
file(GLOB SHADER_FILES "${VOLUME_RENDERER_SRC}*.txt")
file(COPY ${SHADER_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
