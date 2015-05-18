#ifndef CAMERA_H
#define CAMERA_H

#include "GLM.h"

class Camera
{
public:
	glm::vec3 position;
	glm::vec3 focus;
	float distFromFocus;
	float FoV;
	int xPixels, yPixels;

	glm::mat4 viewMat;
	glm::mat4 projMat;

	void Init(int screenWidth, int screenHeight);
	void Update();

	void Zoom(float zoomAmount);
	void Translate(glm::vec3 translateAmount);
	void Rotate(float rotateAmount);
	glm::vec3 GetViewDirection();

	Camera()
	{
		position = glm::vec3(0.0f, 0.0f, 2.0f);
		focus = glm::vec3(0.0f);
		distFromFocus = glm::distance(position, focus);
		xPixels = 800;
		yPixels = 800;
		float nearPlane = 0.1f;
		float farPlane = 1000.0f;
		FoV = 67.0f;
		float aspect = (float)xPixels / (float)yPixels;
		projMat = glm::perspective(FoV, aspect, nearPlane, farPlane);
		viewMat = glm::lookAt(position, focus, glm::vec3(0.0f, 1.0f, 0.0f));
	}

	Camera(int screenWidth, int screenHeight, glm::vec3 _position)
	{
		position = _position;
		focus = glm::vec3(0.0f);
		distFromFocus = glm::distance(position, focus);
		xPixels = screenWidth;
		yPixels = screenHeight;
		float nearPlane = 0.1f;
		float farPlane = 1000.0f;
		FoV = 67.0f;
		float aspect = (float)xPixels / (float)yPixels;
		projMat = glm::perspective(FoV, aspect, nearPlane, farPlane);
		viewMat = glm::lookAt(position, focus, glm::vec3(0.0f, 1.0f, 0.0f));
	}

	void front()
	{
		position = glm::vec3(0.0f, 0.0f, 2.0f);
		focus = glm::vec3(0.0f);
		distFromFocus = glm::distance(position, focus);

		float nearPlane = 0.1f;
		float farPlane = 1000.0f;
		FoV = 67.0f;
		float aspect = (float)xPixels / (float)yPixels;

		projMat = glm::perspective(FoV, aspect, nearPlane, farPlane);
		viewMat = glm::lookAt(position, focus, glm::vec3(0.0f, 1.0f, 0.0f));
	}

	//void top()
	//{
	//	front();
	//	position = glm::rotateX(position, -90.f);
	//}

	//void left()
	//{
	//	front();
	//	position = glm::rotateY(position, -90.f);
	//}

	// Rotate by degrees
	void Camera::rotateX(float rotateAmount)
	{
		position = glm::rotateX(position, rotateAmount);
	}

	// Rotate by degrees
	void Camera::rotateY(float rotateAmount)
	{
		position = glm::rotateY(position, rotateAmount);
	}

	// Rotate by degrees
	void Camera::rotateZ(float rotateAmount)
	{
		position = glm::rotateZ(position, rotateAmount);
	}
};

#endif
