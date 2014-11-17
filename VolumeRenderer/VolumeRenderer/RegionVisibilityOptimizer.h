#ifndef REGION_VISIBILITY_OPTIMIZER_H
#define REGION_VISIBILITY_OPTIMIZER_H

#include <vector>
#include "GLM.h"
#include "ShaderManager.h"
#include "Camera.h"
#include "VolumeDataset.h"
#include "GPURaycaster.h"
#include "TransferFunction.h"

class RegionVisibilityOptimizer
{
public:
	int numRegions;
	std::vector<glm::vec2> regions;

	int xPixels, yPixels;

	GLuint frameBuffer;
	GLuint bufferTex;

	cudaGraphicsResource_t resource;

	float *cudaRegionVisibilities;
	int *cudaNumInRegion;
	std::vector<float> regionVisibilities;

	void Init(TransferFunction &transferFunction);
	void CalculateVisibility(ShaderManager &shaderManager, Camera &camera, VolumeDataset &volume, TransferFunction &transferFunction, Raycaster *raycaster);
};


#endif