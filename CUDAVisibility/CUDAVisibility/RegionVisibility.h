#ifndef REGION_VISIBILITY_H
#define REGION_VISIBILITY_H

#include "VolumeDataset.h"
#include "ShaderManager.h"
#include "CudaHeaders.h"


class RegionVisibility
{
public:

	std::vector<glm::vec2> regions;

	void Init(int screenWidth, int screenHeight);
	void CalculateHistogram(VolumeDataset &volume, GLuint &tfTexture, ShaderManager shaderManager, Camera &camera);

};

#endif