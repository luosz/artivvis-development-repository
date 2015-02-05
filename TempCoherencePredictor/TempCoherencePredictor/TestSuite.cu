#include "TestSuite.h"

void TestSuite::Init(int screenWidth, int screenHeight, VolumeDataset &volume)
{
	errorMetrics.Init(screenWidth, screenHeight);
	changeBetweenFrames.Init(volume);

	
}





void TestSuite::Test(VolumeDataset &volume, TransferFunction &transferFunction, ShaderManager &shaderManager, Camera &camera, Raycaster &raycaster, GLuint bruteTex3D, GLuint interpTex3D, int curretTimestep)
{
	errorMetrics.FindError(transferFunction, shaderManager, camera, raycaster, bruteTex3D, interpTex3D);

//	if (curretTimestep != 0)
//		changeBetweenFrames.Find(volume, curretTimestep, bruteTex3D);
}



TestSuite::~TestSuite()
{
//	errorMetrics.cudaMSE.clear();
//	errorMetrics.cudaMAE.clear();
}





