#include "RegionGrabber.h"

RegionGrabber::RegionGrabber()
{
	cubeWidth = 2;
}

float RegionGrabber::Grab(int mouseX, int mouseY, int screenWidth, int screenHeight, Camera &camera, glm::vec3 clipPlaneNormal, float clipPlaneDistance, VolumeDataset &volume)
{
	glm::vec3 camDirection = camera.GetViewDirection();

	glm::vec3 rightVec = glm::normalize(glm::cross(camDirection, glm::vec3(0.0f, 1.0f, 0.0f)));
	glm::vec3 upVec = glm::normalize(glm::cross(camDirection, -rightVec));

	float extent = glm::tan((camera.FoV / 2.0f) * (glm::pi<float>()/180.0f));

	glm::vec3 topLeft;

	glm::vec3 temp = camera.position + camDirection;
	temp = temp + (extent * upVec);
	topLeft = temp - (extent * rightVec);
	
	float deltaX = (extent * 2.0f) / (float)screenWidth;
	float deltaY = (extent * 2.0f) / (float)screenHeight;

	glm::vec3 mousePos = glm::vec3(topLeft + ((deltaX * mouseX) * rightVec) - ((deltaY * mouseY) * upVec));
	glm::vec3 mouseRayDir = glm::normalize(mousePos - camera.position);



	glm::vec3 planePos(0.0f);

	if (camera.position.z > 0.0f)
		planePos.z = 1.0f - clipPlaneDistance;
	else
		planePos.z = -1.0f + (2.0f - clipPlaneDistance);



	float d = glm::dot(clipPlaneNormal, planePos);

	if (glm::dot(clipPlaneNormal, mouseRayDir) == 0.0f)
	{
		std::cout << "Perpendicular" << std::endl;
		return -1.0f;
	}

	float t = (d - glm::dot(clipPlaneNormal, camera.position)) / glm::dot(clipPlaneNormal, mouseRayDir);
	
	glm::vec3 newRay = mouseRayDir * t;

	glm::vec3 contact = camera.position + newRay;

	if (glm::abs(contact.x >= 1.0f) || glm::abs(contact.y >= 1.0f) || glm::abs(contact.z >= 1.0f))
		return -1.0f;


	int xVoxel, yVoxel, zVoxel, index;

	xVoxel = (int)((contact.x + 1.0f) / (2.0f / volume.xRes));
	yVoxel = (int)((contact.y + 1.0f) / (2.0f / volume.yRes));
	zVoxel = (int)((contact.z + 1.0f) / (2.0f / volume.zRes));

	if (volume.bytesPerElement != 1)
	{
		std::cout << "Only works for byte datasets at the moment" << std::endl;
		return -1.0f;
	}

	float avgIntensity = 0.0f;

	for (int i=-cubeWidth; i<=cubeWidth; i++)
		for (int j=-cubeWidth; j<=cubeWidth; j++)
			for (int k=-cubeWidth; k<=cubeWidth; k++)
			{
				index = (xVoxel + i) + ((yVoxel + j) * volume.xRes) + ((zVoxel + k) * volume.xRes * volume.yRes);
				avgIntensity += (volume.memblock3D[index] / 255.0f);
			}

	avgIntensity /= (glm::pow( (2.0f*cubeWidth + 1.0f), 3.0f));

	std::cout << avgIntensity << std::endl;

	return avgIntensity;
}
