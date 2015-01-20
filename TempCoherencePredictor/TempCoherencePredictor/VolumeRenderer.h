#ifndef VOLUME_RENDERER_H
#define VOLUME_RENDERER_H

#include "Camera.h"
#include "VolumeDataset.h"
#include "ShaderManager.h"
#include "TempCoherence.h"
#include "TransferFunction.h"
#include "Raycaster.h"
#include "BruteForce.h"

class VolumeRenderer
{
public:
	Camera camera;
	GLuint shaderProgramID;
	ShaderManager shaderManager;
	VolumeDataset volume;
	TempCoherence *tempCoherence;
	TransferFunction transferFunction;
	Raycaster *raycaster;
	BruteForce *bruteForce;

	int currentTimestep;
	GLuint tex3D;
	clock_t oldTime;

	void Init(int screenWidth, int screenHeight);
	void Update();

	

};


#endif