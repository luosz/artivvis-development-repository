#ifndef RENDERER_H
#define RENDERER_H

#include "VolumeDataset.h"
#include "GPUContours.h"
#include "GPURaycaster.h"
#include "CPURaycaster.h"
#include "BurnsContours.h"
#include "TransferFunction.h"

class Renderer
{
public:
	Raycaster *raycaster;
	TransferFunction transferFunction;

	virtual void Init(int screenWidth, int screenHeight, VolumeDataset &volume) = 0;
	virtual void Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera) = 0;
};

#endif