#ifndef RENDERER_H
#define RENDERER_H

#include "VolumeDataset.h"
#include "GPUContours.h"
#include "GPURaycaster.h"
#include "CPURaycaster.h"
#include "BurnsContours.h"
#include "TransferFunction.h"
#include "TFOptimizer.h"
#include "VisibilityHistogram.h"

class Renderer
{
public:
	Raycaster *raycaster;
	TransferFunction transferFunction;
	VisibilityHistogram visibilityHistogram;
	TFOptimizer *optimizer;

	virtual void Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera) = 0;
};

#endif