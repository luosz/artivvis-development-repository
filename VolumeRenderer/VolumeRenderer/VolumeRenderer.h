#ifndef VOLUME_RENDERER_H
#define VOLUME_RENDERER_H

#include "Camera.h"
#include "VolumeDataset.h"
#include "ShaderManager.h"
#include "OpenGLRenderer.h"
#include "MarchingCubesRenderer.h"
#include "CPURenderer.h"

class VolumeRenderer
{
public:
	Camera camera;
	GLuint shaderProgramID;
	
	ShaderManager shaderManager;
	VolumeDataset volume;
	Renderer *renderer;

	void Init(int screenWidth, int screenHeight);
	void Update();
};


#endif