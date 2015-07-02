#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include "GLM.h"
#include "Framebuffer.h"
#include "CudaHeaders.h"
#include "Texture2D.h"
#include <thrust\reduce.h>
#include <thrust\device_vector.h>

#define A_SQ_DIFF_RADIUS 2
#define B_SQ_DIFF_RADIUS 5

class ImageProcessor
{
public:
	Framebuffer *framebuffer;
	Texture2D *fbTex;
	int numPixels, xPixels, yPixels;
	thrust::device_vector<float> aSquareDiffs, bSquareDiffs, pixelVals;
	cudaGraphicsResource_t cudaResource;
	int kernelRadius;
	float *cudaPixelMean;

	void Init(int screenWidth, int screenHeight);
	void Begin();
	void GetAutoCorrelation();
	void End();

	void WriteToTexture();
};


#endif