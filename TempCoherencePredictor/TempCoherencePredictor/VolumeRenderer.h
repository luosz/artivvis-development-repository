#ifndef VOLUME_RENDERER_H
#define VOLUME_RENDERER_H

#include "Camera.h"
#include "VolumeDataset.h"
#include "ShaderManager.h"
#include "TempCoherence.h"
#include "TransferFunction.h"
#include "Raycaster.h"
#include "BruteForce.h"
#include "TestSuite.h"
#include "WriteToFile.h"

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
	TestSuite tester;
	WriteToFile fileWriter;

	bool writeToFile;
	

	int currentTimestep;
	GLuint bruteTex3D, interpTex3D;
	clock_t oldTime;

	void Init(int screenWidth, int screenHeight);
	void Update();
};


#endif