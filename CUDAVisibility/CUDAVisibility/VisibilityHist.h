#ifndef VISIBILITY_HIST_H
#define VISIBILITY_HIST_H

#include "CudaHeaders.h"
#include "VolumeDataset.h"
#include "TransferFunction.h"
#include "ShaderManager.h"

class VisibilityHistogram
{
public:
	float* pixelBuffer;

	int xPixels, yPixels;
	std::vector<GLuint> slices;
	int numSlices;
	std::vector<glm::vec3> boxCorners;
	GLuint frameBuffer;
	GLuint opacityTex;
	int currentSlice;

	std::vector<float> visibilities;
	std::vector<int> numVis;


	void Init(int screenWidth, int screenHeight);
	void CalculateHistogram(VolumeDataset &volume, TransferFunction &transferFunction, ShaderManager shaderManager, Camera &camera);
	void DrawHistogram(ShaderManager shaderManager, Camera & camera);

	void InitSlices();
	GLuint GenerateSliceTexture();
	glm::vec3 FindClosestCorner(Camera &camera);
};

#endif