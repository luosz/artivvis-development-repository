#ifndef ERROR_METRICS_H
#define ERROR_METRICS_H

#include "ShaderManager.h"
#include "CudaHeaders.h"
#include "Raycaster.h"
#include "Camera.h"
#include "TransferFunction.h"
#include "Framebuffer.h"
#include "thrust\reduce.h"
#include "thrust\device_ptr.h"
#include <thrust/device_vector.h>
#include <thrust\count.h>
#include <math.h>

class ErrorMetrics
{
public:
	int xPixels, yPixels, numPixels;
	GLuint interpImage, bruteImage;
	Framebuffer framebuffer;

	float meanSqrError, meanAvgErr, laplaceMSE, peakSigToNoise, maxDifference;
	int *cudaNumZeroLaplacians;
	thrust::device_vector<float> cudaMSE, cudaMAE, cudaLMSE;
	std::vector<cudaGraphicsResource_t> cudaResources;

	void Init(int screenWidth, int screenHeight);
	GLuint Generate2DTexture();
	void FindError(TransferFunction &transferFunction, ShaderManager &shaderManager, Camera &camera, Raycaster &raycaster, GLuint bruteTex, GLuint interpTex);
	void RenderImages(TransferFunction &transferFunction, ShaderManager &shaderManager, Camera &camera, Raycaster &raycaster, GLuint bruteTex, GLuint interpTex);
	void CompareImages();
	void GetErrorMetrics();

	~ErrorMetrics();
};

#endif