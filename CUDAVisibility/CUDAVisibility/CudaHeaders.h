#ifndef CUDA_HEADERS_H
#define CUDA_HEADERS_H

#define GLM_FORCE_CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaGL.h"
#include "cuda_gl_interop.h"
#include <iostream>
#include "GLM.h"

static void HandleError(cudaError_t err, const char *file, int line ) 
{
    if (err != cudaSuccess) 
	{
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
		getchar();
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#endif