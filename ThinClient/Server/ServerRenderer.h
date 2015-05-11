#ifndef SERVER_RENDERER_H
#define SERVER_RENDERER_H

#include "Camera.h"
#include "ShaderManager.h"
#include "TransferFunction.h"
#include "Raycaster.h"

class VolumeRenderer
{
public:
	Camera camera;
	GLuint shaderProgramID;
	VolumeDataset *volume;
	TransferFunction transferFunction;
	Raycaster *raycaster;

	void Init(int screenWidth, int screenHeight, VolumeDataset &volume_);
	void Update();
};


#endif