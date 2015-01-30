#ifndef VISIBILITY_HIST_H
#define VISIBILITY_HIST_H

#include "VolumeDataset.h"
#include "ShaderManager.h"
#include "CudaHeaders.h"


class VisibilityHistogram
{
public:
	float* pixelBuffer;

	int xPixels, yPixels;
	std::vector<GLuint> slices;
	int numSlices;
	int numBins;
	std::vector<glm::vec3> boxCorners;
	GLuint frameBuffer;
	GLuint opacityTex1, opacityTex2;
	int currentSlice;
	
	bool grabFrustum;
	int mousePosX, mousePosY;
	int frustumExtent;

	std::vector<float> visibilities;
	std::vector<int> numVis;
	std::vector<int> intensity_histogram;

	cudaGraphicsResource_t resource;

	float *cudaHistBins;
	int *cudaNumInBin;
	int *cudaNumInBin_intensity;

	void Init(int screenWidth, int screenHeight);
	void CalculateHistogram(VolumeDataset &volume, GLuint &tfTexture, ShaderManager shaderManager, Camera &camera);
	void DrawHistogram(ShaderManager shaderManager, Camera & camera);

	GLuint GenerateSliceTexture();
	glm::vec3 FindClosestCorner(Camera &camera);
	glm::vec3 FindFarthestCorner(Camera &camera);

};

#endif