#ifndef CONTOUR_DRAWER_H
#define CONTOUR_DRAWER_H

#include "ShaderManager.h"
#include "Camera.h"
#include "VolumeDataset.h"
#include "Raycaster.h"

class ContourDrawer
{
public:
	int numPixelsLower;
	float suggestiveContourThreshold;
	int kernelRadius;

	bool compute;

	virtual void Init(int screenWidth, int screenHeight, VolumeDataset &volume) = 0;
	virtual void DrawContours(VolumeDataset &volume, Camera &camera, ShaderManager &shaderManager, Raycaster &raycaster) = 0;
};

#endif



	