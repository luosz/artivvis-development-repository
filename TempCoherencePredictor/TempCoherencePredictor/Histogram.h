#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <vector>
#include "TransferFunction.h"
#include "ShaderManager.h"
#include "VolumeDataset.h"
#include "Camera.h"

class Histogram
{
public:
	int numBins;
	std::vector<float> values;

	virtual void Update(int currentTimestep, VolumeDataset &volume, GLuint tex3D, GLuint &tfTexture, ShaderManager &shaderManager, Camera &camera) = 0;
};

#endif