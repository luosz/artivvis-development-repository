#ifndef VISIBILITY_HIST_H
#define VISIBILITY_HIST_H

#include "VolumeDataset.h"
#include "ShaderManager.h"
#include "CudaHeaders.h"
#include "Framebuffer.h"
#include "Histogram.h"


class VisibilityHistogram  :  public Histogram
{
public:
	int xPixels, yPixels;
	std::vector<GLuint> slices;
	int numSlices;
	std::vector<glm::vec3> boxCorners;
	Framebuffer frameBuffer;
	GLuint opacityTex1, opacityTex2;
	int currentSlice;
	
	bool grabFrustum;
	int mousePosX, mousePosY;
	int frustumExtent;

	std::vector<int> numVis;

	cudaGraphicsResource_t resource;

	float *cudaHistBins;
	int *cudaNumInBin;


	VisibilityHistogram(int screenWidth, int screenHeight);
	void Update(int currentTimestep, VolumeDataset &volume, GLuint tex3D, GLuint &tfTexture, ShaderManager &shaderManager, Camera &camera);

	void DrawHistogram(ShaderManager &shaderManager, Camera & camera);

	GLuint GenerateSliceTexture();
	glm::vec3 FindClosestCorner(Camera &camera);
	glm::vec3 FindFarthestCorner(Camera &camera);

};

#endif