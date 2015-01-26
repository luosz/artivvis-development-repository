#ifndef REGION_GRABBER_H
#define REGION_GRABBER_H

#include "GLM.h"
#include "VolumeDataset.h"
#include "Camera.h"
#include "VisibilityHistogram.h"

class RegionGrabber
{
public:
	int cubeWidth;

	RegionGrabber();
	float Grab(int mouseX, int mouseY, int screenWidth, int screenHeight, Camera &camera, glm::vec3 clipPlaneNormal, float clipPlaneDistance, VolumeDataset &volume);

	std::vector<float> GrabVisibility(int mouseX, int mouseY, int screenWidth, int screenHeight, Camera &camera, glm::vec3 clipPlaneNormal, float clipPlaneDistance, VolumeDataset &volume, const VisibilityHistogram &visibilityHistogram);

	/// Average intensity over adjacent voxels
	float getAverageIntensity(int xVoxel, int yVoxel, int zVoxel, const VolumeDataset &volume)
	{
		float avgIntensity = 0.0f;
		for (int i=-cubeWidth; i<=cubeWidth; i++)
			for (int j=-cubeWidth; j<=cubeWidth; j++)
				for (int k=-cubeWidth; k<=cubeWidth; k++)
				{
					auto index = (xVoxel + i) + ((yVoxel + j) * volume.xRes) + ((zVoxel + k) * volume.xRes * volume.yRes);
					avgIntensity += (volume.memblock3D[index] / 255.0f);
				}
		avgIntensity /= (glm::pow( (2.0f*cubeWidth + 1.0f), 3.0f));
		std::cout << avgIntensity << std::endl;
		return avgIntensity;
	}

	float averageIntensityWithVisibility(int xVoxel, int yVoxel, int zVoxel, const VolumeDataset &volume, const VisibilityHistogram &visibilityHistogram)
	{
		// Average intensity over adjacent voxels
		float avgIntensity = 0.0f;
		for (int i = -cubeWidth; i <= cubeWidth; i++)
			for (int j = -cubeWidth; j <= cubeWidth; j++)
				for (int k = -cubeWidth; k <= cubeWidth; k++)
				{
					auto index = (xVoxel + i) + ((yVoxel + j) * volume.xRes) + ((zVoxel + k) * volume.xRes * volume.yRes);
					auto intensity = volume.memblock3D[index];
					auto visibility = visibilityHistogram.visibilities[intensity];
					avgIntensity += (intensity * visibility / 255.0f);
				}
		avgIntensity /= (glm::pow((2.0f*cubeWidth + 1.0f), 3.0f));
		std::cout << avgIntensity << std::endl;
		return avgIntensity;
	}
};

#endif
