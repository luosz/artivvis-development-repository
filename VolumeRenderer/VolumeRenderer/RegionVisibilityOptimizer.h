#ifndef REGION_VISIBILITY_OPTIMIZER_H
#define REGION_VISIBILITY_OPTIMIZER_H

#include <vector>
#include "GLM.h"
#include "TFOptimizer.h"
#include "CudaHeaders.h"

class RegionVisibilityOptimizer     :     public TFOptimizer
{
public:
	RegionVisibilityOptimizer(VolumeDataset *volume_, TransferFunction *transferFunction_, Raycaster *raycaster_, ShaderManager *shaderManager_, Camera *camera_);
	~RegionVisibilityOptimizer();
	void Optimize();
	void Draw(ShaderManager &shaderManager, Camera &camera);


	VolumeDataset *volume;
	TransferFunction *transferFunction;
	Raycaster *raycaster;	
	ShaderManager *shaderManager;
	Camera *camera;

	int numRegions;
	std::vector<glm::vec2> regions;
	int xPixels, yPixels;

	GLuint frameBuffer;
	GLuint renderBuffer;
	GLuint bufferTex;

	cudaGraphicsResource_t resource;

	float *cudaRegionVisibilities;
	int *cudaNumInRegion;
	std::vector<float> regionVisibilities;
	void CalculateVisibility();
	
	
};


#endif