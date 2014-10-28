#include "Camera.h"

void Camera::Init(int screenWidth, int screenHeight)
{
	position =  glm::vec3(0.0f, 0.0f, 3.0f);
	focus = glm::vec3(0.0f);
	distFromFocus = glm::distance(position, focus);

	float nearPlane = 0.1f;
	float farPlane = 100.0f;
	FoV = 67.0f;
	float aspect = screenWidth / screenHeight;

	projMat = glm::perspective(FoV, aspect, nearPlane, farPlane);

	viewMat = glm::lookAt(position, focus, glm::vec3(0.0f,1.0f,0.0f));
}




void Camera::Update()
{
	viewMat = glm::lookAt(position, focus, glm::vec3(0.0f,1.0f,0.0f));
}
