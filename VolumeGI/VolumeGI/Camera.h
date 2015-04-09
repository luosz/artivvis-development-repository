#ifndef CAMERA_H
#define CAMERA_H

#include "GLM.h"


class Camera
{
public:
	glm::vec3 position;
	glm::vec3 focus;
	float FoV;
	int xPixels, yPixels;
	float rotationSpeed;
	float translateSpeed;
	float distFromFocus;

	glm::mat4 viewMat;
	glm::mat4 projMat;

	void Init(int screenWidth, int screenHeight);
	void Init(int xPixels_, int yPixels_, float FoV_, glm::vec3 pos, glm::vec3 dir);
	void InitOrtho(int xPixels_, int yPixels_, int width, int height, glm::vec3 pos, glm::vec3 lookDir);
	void Update();

	void TranslateForwards(float translateAmount);
	void TranslateSideways(float translateAmount);
	void TranslateUp(float translateAmount);
	void XRotate(float rotateAmount);
	void YRotate(float rotateAmount);
	void OrbitSideways(float rotateAmount);
	void OrbitUp(float translateAmount);
	void OrbitZoom(float zoomAmount);
};


#endif