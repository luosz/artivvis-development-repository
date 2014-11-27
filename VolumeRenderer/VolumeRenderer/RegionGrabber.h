#ifndef REGION_GRABBER_H
#define REGION_GRABBER_H

#include "GLM.h"
#include "VolumeDataset.h"
#include "Camera.h"

class RegionGrabber
{
public:
	int cubeWidth;

	RegionGrabber();
	float Grab(int mouseX, int mouseY, int screenWidth, int screenHeight, Camera &camera, glm::vec3 clipPlaneNormal, float clipPlaneDistance, VolumeDataset &volume);
};

#endif