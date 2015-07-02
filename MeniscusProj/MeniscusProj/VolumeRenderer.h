#ifndef VOLUME_RENDERER_H
#define VOLUME_RENDERER_H

#include "Camera.h"
#include "VolumeDataset.h"
#include "ShaderManager.h"
#include "OpenGLRenderer.h"
#include <time.h>
#include <iomanip>

class VolumeRenderer
{
public:
	Camera camera;
	ShaderManager shaderManager;
	VolumeDataset volume;
	OpenGLRenderer *renderer;
	ClipPlane clipPlane;

	int numRemoved, numInFocused;

	GLubyte *tempVol;
	int tempXRes, tempYRes, tempZRes;

	bool focused, removed;
	float sphereRadius;
	glm::vec3 spherePoint;

	std::vector<glm::vec3> boxPoints;

	clock_t oldTime;
	int currentTimestep;

	void Init(int screenWidth, int screenHeight);
	void Update();

	void DrawBox();
	void AddBoxPoint(int xMousePos, int yMousePos);
	void FocusVolume();

	void FindRemoved();
	void DrawMyText();

	void Reset();
};


#endif