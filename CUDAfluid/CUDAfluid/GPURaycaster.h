#ifndef GPU_RAYCASTER_H
#define GPU_RAYCASTER_H

#include <GL/glew.h>
#include <GL/freeglut.h>
#include "Camera.h"
#include "GLM.h"
#include <vector>
#include "Constants.h"


class GPURaycaster
{
public:
	GLuint texID, tex2ID;

	int maxRaySteps;
	float rayStepSize;
	float gradientStepSize;

	void Init(int screenWidth, int screenHeight);
	void GenerateTexture();
	void Raycast(GLuint shaderProgramID, Camera &camera, std::vector<float> &vector1, std::vector<float> &vector2);
};

#endif

