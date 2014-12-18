cmake_minimum_required(VERSION 2.8.11)

project("VolumeRenderer")

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

set(VOLUME_RENDERER_SRC "VolumeRenderer/" CACHE STRING "Volume Renderer source code directory." FORCE)
set(GLM_SRC "gl/glm/" CACHE STRING "Directory containing source headers for the GLM." FORCE)
set(DEVIL_SRC "gl/include/IL/" CACHE STRING "Directory containing source headers for the DevIL library.")
set(INCLUDE_DIR "gl/include" CACHE STRING "Directory containing source headers for the additional includes." FORCE)
set(CUDA_INCLUDE_DIR "/usr/local/cuda-6.5/include/" CACHE STRING "Include directory for the CUDA toolkit.")
set(QT_INSTALL_PREFIX "" CACHE STRING "Include directory for the Qt libraries.")

set(CMAKE_PREFIX_PATH ${QT_INSTALL_PREFIX})

find_package(OpenGL REQUIRED)
find_package(Qt5Widgets REQUIRED)

include_directories(${GLM_SRC})
include_directories(${DEVIL_SRC})
include_directories(${INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIR})
include_directories(${Qt5Widgets_INCLUDES_DIR})
include_directories(${VOLUME_RENDERER_SRC}GeneratedFiles/)

add_definitions( -Wfatal-errors )

add_executable(VolumeRenderer ${VOLUME_RENDERER_SRC}main.cpp)

target_link_libraries(VolumeRenderer Qt5::Widgets)
