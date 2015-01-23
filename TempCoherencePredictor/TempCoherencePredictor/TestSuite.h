#ifndef TEST_SUITE_H
#define TEST_SUITE_H

#include "ShaderManager.h"
#include "CudaHeaders.h"
#include "Raycaster.h"
#include "VolumeDataset.h"
#include "Camera.h"
#include "TransferFunction.h"
#include "ErrorMetrics.h"
#include "ChangeBetweenFrames.h"

class TestSuite
{
	
public:
	ErrorMetrics errorMetrics;
	ChangeBetweenFrames changeBetweenFrames;

	void Init(int screenWidth, int screenHeight, VolumeDataset &volume);	
	void Test(VolumeDataset &volume, TransferFunction &transferFunction, ShaderManager &shaderManager, Camera &camera, Raycaster &raycaster, GLuint bruteTex, GLuint interpTex, int currentTimestep);

	~TestSuite();
	
};

#endif