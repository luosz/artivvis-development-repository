#ifndef VOLUME_RENDERER_H
#define VOLUME_RENDERER_H

#include "Camera.h"
#include "VolumeDataset.h"
#include "ShaderManager.h"
#include "OpenGLRenderer.h"

class VolumeRenderer
{
public:
	Camera camera;
	GLuint shaderProgramID;
	ShaderManager shaderManager;
	VolumeDataset volume;
	OpenGLRenderer *renderer;

	void Init(int screenWidth, int screenHeight);
	void Update();

};


#endif