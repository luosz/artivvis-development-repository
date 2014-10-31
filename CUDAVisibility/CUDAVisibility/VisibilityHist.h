#ifndef VISIBILITY_HIST_H
#define VISIBILITY_HIST_H

#include "CudaHeaders.h"
#include "VolumeDataset.h"
#include "TransferFunction.h"
#include "ShaderManager.h"

class VisibilityHistogram
{
public:
	char* cudaTexture;

	int xPixels, yPixels;

	void Init(int screenWidth, int screenHeight, VolumeDataset &volume);
	void CalculateHistogram(VolumeDataset &volume, TransferFunction &transferFunction, Camera &camera, ShaderManager shaderManager);
};

#endif