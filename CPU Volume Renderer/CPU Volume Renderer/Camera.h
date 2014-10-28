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

	glm::mat4 viewMat;
	glm::mat4 projMat;

	void Init(int screenWidth, int screenHeight);
	void Update();

};


#endif