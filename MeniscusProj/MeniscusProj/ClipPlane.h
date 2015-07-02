#ifndef CLIP_PLANE_H
#define CLIP_PLANE_H

#include "GLM.h"

class ClipPlane
{
public:
	bool active;
	glm::vec3 point;
	glm::vec3 normal;

	ClipPlane()
	{
		active = false;
		point = glm::vec3(0.0f, 0.0f, 1.0f);
		normal = glm::vec3(0.0f, 0.0f, 1.0f);
	}

	void Move(float x)
	{
		point += x * normal;

		point.x = glm::clamp(point.x, -1.0f, 1.0f);
		point.y = glm::clamp(point.y, -1.0f, 1.0f);
		point.z = glm::clamp(point.z, -1.0f, 1.0f);
	}

	glm::vec3 Intersect(int mouseX, int mouseY, Camera &camera)
	{
		// Find position of mouse in 3D space by colliding it with a plane perpendicular to the camera and then cast a ray from the camera through that position
		glm::vec3 camDirection = camera.GetViewDirection();

		glm::vec3 rightVec = glm::normalize(glm::cross(camDirection, glm::vec3(0.0f, 1.0f, 0.0f)));
		glm::vec3 upVec = glm::normalize(glm::cross(camDirection, -rightVec));

		float yExtent = glm::tan((camera.FoV / 2.0f) * (glm::pi<float>()/180.0f));
		float xExtent = yExtent * camera.aspectRatio;

		glm::vec3 topLeft;

		glm::vec3 temp = camera.position + camDirection;
		temp = temp + (yExtent * upVec);
		topLeft = temp - (xExtent * rightVec);
		
		float deltaX = (xExtent * 2.0f) / (float)camera.xPixels;
		float deltaY = (yExtent * 2.0f) / (float)camera.yPixels;

		glm::vec3 mousePos = glm::vec3(topLeft + ((deltaX * mouseX) * rightVec) - ((deltaY * mouseY) * upVec));
		glm::vec3 mouseRayDir = glm::normalize(mousePos - camera.position);

		// Find where mouse ray collides with clipping plane
		float d = glm::dot(normal, point);

		if (glm::dot(normal, mouseRayDir) == 0.0f)
		{
			std::cout << "Perpendicular" << std::endl;
			return glm::vec3(0.0f);
		}

		float t = (d - glm::dot(normal, camera.position)) / glm::dot(normal, mouseRayDir);
		
		glm::vec3 newRay = mouseRayDir * t;

		glm::vec3 contact = camera.position + newRay;

		// Check if collision falls outside of volume
		if (glm::abs(contact.x >= 1.0f) || glm::abs(contact.y >= 1.0f) || glm::abs(contact.z >= 1.0f))
			return glm::vec3(0.0f);

		return contact;
	}
};

#endif