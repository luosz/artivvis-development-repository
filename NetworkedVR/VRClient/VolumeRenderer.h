#ifndef VOLUME_RENDERER_H
#define VOLUME_RENDERER_H

#include "Camera.h"
#include "ShaderManager.h"
#include "TransferFunction.h"
#include "Raycaster.h"
#include "ImageProcessor.h"

class VolumeRenderer
{
public:
	Camera camera;
	GLuint shaderProgramID;
	VolumeDataset volume;
	TransferFunction transferFunction;
	Raycaster *raycaster;
	ImageProcessor imageProcessor;
	
	GLuint bruteTex3D, interpTex3D;

	void Init(int screenWidth, int screenHeight);
	void Update();
};


#endif