#include "Camera.h"

// Initialize view, projection and position
void Camera::Init(int screenWidth, int screenHeight)
{
	position =  glm::vec3(0.0f, 0.0f, 2.5f);
	focus = glm::vec3(0.0f);
	distFromFocus = glm::distance(position, focus);
	xPixels = screenWidth;
	yPixels = screenHeight;
	rotationSpeed = 0.01f;
	translateSpeed = 0.5f;

	float nearPlane = 0.1f;
	float farPlane = 1000.0f;
	FoV = 67.0f;

	float aspect = (float)xPixels / (float)yPixels;

	projMat = glm::perspective(FoV, aspect, nearPlane, farPlane);

	viewMat = glm::lookAt(position, focus, glm::vec3(0.0f,1.0f,0.0f));
}

void Camera::Init(int xPixels_, int yPixels_, float FoV_, glm::vec3 pos, glm::vec3 dir)
{
	position = pos;
	focus = position + glm::normalize(dir);
	distFromFocus = glm::distance(position, focus);
	xPixels = xPixels_;
	yPixels = yPixels_;
	rotationSpeed = 0.01f;
	translateSpeed = 0.5f;

	float nearPlane = 0.1f;
	float farPlane = 1000.0f;
	FoV = FoV_;

	float aspect = (float)xPixels / (float)yPixels;

	projMat = glm::perspective(FoV, aspect, nearPlane, farPlane);

	viewMat = glm::lookAt(position, focus, glm::vec3(0.0f,1.0f,0.0f));
}

void Camera::InitOrtho(int xPixels_, int yPixels_, int width, int height, glm::vec3 pos, glm::vec3 lookDir)
{
	position =  pos;
	focus = position + lookDir;
	xPixels = xPixels_;
	yPixels = yPixels_;

	float nearPlane = 0.1f;
	float farPlane = 100.0f;

	float left = -width;
	float right = +width;
	float bottom = -height;
	float top = +height;

	glm::vec3 up(0.0f, 1.0f, 0.0);

	if (glm::abs(lookDir.y) == 1.0f)
		up = glm::vec3(0.0f, 0.0f, -1.0f);

	projMat = glm::ortho(left, right, bottom, top, nearPlane, farPlane);

	viewMat = glm::lookAt(position, focus, up);
}


// Update View Matrix
void Camera::Update()
{
	viewMat = glm::lookAt(position, focus, glm::vec3(0.0f,1.0f,0.0f));
}


void Camera::TranslateUp(float translateAmount)
{
	translateAmount *= translateSpeed;

	position.y += translateAmount;
	focus.y += translateAmount;
}


void Camera::TranslateForwards(float translateAmount)
{
	translateAmount *= translateSpeed;

	glm::vec3 viewVec = glm::normalize(focus - position);

	position += viewVec * translateAmount;
	focus += viewVec * translateAmount;
}

void Camera::TranslateSideways(float translateAmount)
{
	translateAmount *= translateSpeed;

	glm::vec3 viewVec = glm::normalize(focus - position);
	glm::vec3 perpVec = glm::cross(viewVec, glm::vec3(0.0f, 1.0f, 0.0f));

	position += perpVec * translateAmount;
	focus += perpVec * translateAmount;
}


// Rotate by degrees
void Camera::XRotate(float rotateAmount)
{
	rotateAmount *= rotationSpeed;


	glm::vec3 viewVec = focus - position;

	glm::mat3 xRotMat = glm::mat3(	glm::cos(rotateAmount), 0.0f, glm::sin(rotateAmount),
									0.0f, 1.0f, 0.0f,	
									-glm::sin(rotateAmount), 0.0f, glm::cos(rotateAmount));

	viewVec = xRotMat * viewVec;

	focus = position + glm::normalize(viewVec);
}


void Camera::YRotate(float rotateAmount)
{
	rotateAmount *= rotationSpeed;
	focus += glm::vec3(0.0f, rotateAmount, 0.0f);

	focus = position + glm::normalize(focus - position);
}

void Camera::OrbitSideways(float rotateAmount)
{
	position = glm::rotateY(position, rotateAmount);
}

void Camera::OrbitUp(float translateAmount)
{
	position += glm::vec3(0.0f, translateAmount, 0.0f);

	position = focus + (glm::normalize(position - focus) * distFromFocus);
}

void Camera::OrbitZoom(float zoomAmount)
{
	distFromFocus += zoomAmount;

	position = focus + (glm::normalize(position - focus) * distFromFocus);
}