#ifndef VOLUME_RENDERER_H
#define VOLUME_RENDERER_H

#include "Camera.h"
#include "VolumeDataset.h"
#include "ShaderManager.h"
#include "OpenGLRenderer.h"
#include "MarchingCubesRenderer.h"
#include "CPURenderer.h"
#include "RegionGrabber.h"
#include "TomsOGLRenderer.h"
#include "JoesOGLRenderer.h"

// Include use_JoesOGLRenderer.h to enable Joe's OpenGL Renderer, otherwise Tom's OpenGL renderer is used.
#include "use_JoesOGLRenderer.h"

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

	void OptimizeForSelectedRegionWithVisibility(int mouseX, int mouseY, int screenWidth, int screenHeight)
	{
		float avgIntensity = regionGrabber.Grab(mouseX, mouseY, screenWidth, screenHeight, camera, renderer->raycaster->clipPlaneNormal, renderer->raycaster->clipPlaneDistance, volume);

		if (avgIntensity == -1.0f)
			return;

		renderer->transferFunction.targetIntensity = avgIntensity;
	}
};


#endif