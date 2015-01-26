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
#include "RegionVisibilityOptimizer.h"
#include "VisibilityTFOptimizer.h"
#include "IntensityTFOptimizer.h"
#include "IntensityTFOptimizerV2.h"

class Renderer
{
public:
	TransferFunction transferFunction;
	VisibilityHistogram visibilityHistogram;
	Raycaster *raycaster;
	TFOptimizer *_optimizer;
	IntensityTFOptimizerV2 *_intensityTFOptimizerV2;

	Renderer()
	{
		std::cout << "Renderer is created" << std::endl;
		raycaster = NULL;
		_optimizer = NULL;
		_intensityTFOptimizerV2 = NULL;
	}

	virtual void Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera) = 0;

	virtual TFOptimizer * optimizer()
	{ 
		if (!_optimizer)
		{
			std::cout << "Warning: _optimizer is NULL" << std::endl;
		}
		return _optimizer; 
	}

	virtual IntensityTFOptimizerV2 * intensityTFOptimizerV2()
	{ 
		if (!_intensityTFOptimizerV2)
		{
			std::cout << "Warning: _intensityTFOptimizerV2 is NULL" << std::endl;
		}
		return _intensityTFOptimizerV2;
	}
};

#endif
