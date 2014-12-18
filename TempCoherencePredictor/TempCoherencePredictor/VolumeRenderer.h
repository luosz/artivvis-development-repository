#ifndef VOLUME_RENDERER_H
#define VOLUME_RENDERER_H

#include "Camera.h"
#include "VolumeDataset.h"
#include "ShaderManager.h"
#include "BlockRaycaster.h"
#include "TransferFunction.h"

class VolumeRenderer
{
public:
	Camera camera;
	GLuint shaderProgramID;
	ShaderManager shaderManager;
	VolumeDataset volume;
	BlockRaycaster *raycaster;
	TransferFunction transferFunction;

	void Init(int screenWidth, int screenHeight);
	void Update();

};


#endif