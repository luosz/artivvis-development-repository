#ifndef VOLUME_RENDERER_H
#define VOLUME_RENDERER_H

#include "Camera.h"
#include "VolumeDataset.h"
#include "ShaderManager.h"
#include "OpenGLRenderer.h"
#include "MarchingCubesRenderer.h"
#include "CPURenderer.h"
#include "RegionGrabber.h"

class VolumeRenderer
{
public:
	Camera camera;
	GLuint shaderProgramID;
	ShaderManager shaderManager;
	VolumeDataset volume;
	Renderer *renderer;
	RegionGrabber regionGrabber;

	bool grabRegion;

	void Init(int screenWidth, int screenHeight);
	void Update();

	void OptimizeForSelectedRegion(int mouseX, int mouseY, int screenWidth, int screenHeight);
};


#endif