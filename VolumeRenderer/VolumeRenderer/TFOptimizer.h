#ifndef TF_OPTIMIZER_H
#define TF_OPTIMIZER_H

#include "ShaderManager.h"
#include "Camera.h"
#include "VolumeDataset.h"
#include "GPURaycaster.h"
#include "TransferFunction.h"
#include "VisibilityHistogram.h"

class TFOptimizer
{
public:
	virtual void Optimize() = 0;
	virtual void Draw(ShaderManager &shaderManager, Camera &camera) { }
};


#endif