#ifndef BLOCK_RAYCASTER_H
#define BLOCK_RAYCASTER_H

#include <GL/glew.h>
#include <GL/freeglut.h>
#include "Camera.h"
#include "GLM.h"
#include <vector>
#include "VolumeDataset.h"
#include "TransferFunction.h"
#include "CudaHeaders.h"

#define EXTRAP_CONST 2

struct Block
{
	int blockRes;
	int xIndex, yIndex, zIndex;
	std::vector<glm::vec3> vertices;

	Block(int blockRes_, int xID, int yID, int zID, float xVoxelWidth, float yVoxelWidth, float zVoxelWidth)
	{
		blockRes = blockRes_;
		xIndex = xID;
		yIndex = yID;
		zIndex = zID;

		vertices.reserve(8);

		// Bottom
		vertices.push_back(glm::vec3(-1.0f + (xID * xVoxelWidth * blockRes),								-1.0f + (yID * yVoxelWidth * blockRes),								-1.0f + (zID * zVoxelWidth * blockRes)));
		vertices.push_back(glm::vec3(-1.0f + (xID * xVoxelWidth * blockRes),								-1.0f + (yID * yVoxelWidth * blockRes),								-1.0f + (zID * zVoxelWidth * blockRes) + (zVoxelWidth * blockRes)));
		vertices.push_back(glm::vec3(-1.0f + (xID * xVoxelWidth * blockRes) + (xVoxelWidth * blockRes),		-1.0f + (yID * yVoxelWidth * blockRes),								-1.0f + (zID * zVoxelWidth * blockRes)));
		vertices.push_back(glm::vec3(-1.0f + (xID * xVoxelWidth * blockRes) + (xVoxelWidth * blockRes),		-1.0f + (yID * yVoxelWidth * blockRes),								-1.0f + (zID * zVoxelWidth * blockRes) + (zVoxelWidth * blockRes)));

		// Top
		vertices.push_back(glm::vec3(-1.0f + (xID * xVoxelWidth * blockRes),								-1.0f + (yID * yVoxelWidth * blockRes) + (yVoxelWidth * blockRes),	-1.0f + (zID * zVoxelWidth * blockRes)));
		vertices.push_back(glm::vec3(-1.0f + (xID * xVoxelWidth * blockRes),								-1.0f + (yID * yVoxelWidth * blockRes) + (yVoxelWidth * blockRes),	-1.0f + (zID * zVoxelWidth * blockRes) + (zVoxelWidth * blockRes)));
		vertices.push_back(glm::vec3(-1.0f + (xID * xVoxelWidth * blockRes) + (xVoxelWidth * blockRes),		-1.0f + (yID * yVoxelWidth * blockRes) + (yVoxelWidth * blockRes),	-1.0f + (zID * zVoxelWidth * blockRes)));
		vertices.push_back(glm::vec3(-1.0f + (xID * xVoxelWidth * blockRes) + (xVoxelWidth * blockRes),		-1.0f + (yID * yVoxelWidth * blockRes) + (yVoxelWidth * blockRes),	-1.0f + (zID * zVoxelWidth * blockRes) + (zVoxelWidth * blockRes)));
	}
};

class BlockRaycaster
{
public:
	int maxRaySteps;
	float rayStepSize;
	float gradientStepSize;
	glm::vec3 lightPosition;

	std::vector<Block> blocks;
	int blockRes;
	int numBlocks;
	int numXBlocks, numYBlocks, numZBlocks;

	int textureSize;
	GLuint currTexture3D;
	GLuint prevTexture3D;
	GLuint nextTexture3D;
	unsigned char *prevTempVolume;
	unsigned char *currTempVolume;
	unsigned char *nextTempVolume;
	int epsilon;
	float extrapConst;
	int numBlocksCopied, numBlocksExtrapolated;

	int currentTimestep;
	clock_t oldTime;

	std::vector<cudaGraphicsResource_t> cudaResources;
	bool *cudaBlockNeedsCopy;

	void TemporalCoherence(VolumeDataset &volume);
	void GPUPredict(VolumeDataset &volume);
	void CPUPredict(VolumeDataset &volume);
	bool BlockCompare(VolumeDataset &volume, int x, int y, int z);

	BlockRaycaster(int screenWidth, int screenHeight, VolumeDataset &volume);
	void Raycast(VolumeDataset &volume, TransferFunction &transferFunction, GLuint shaderProgramID, Camera &camera);
	
	void BlockRaycast(VolumeDataset &volume, TransferFunction &transferFunction, GLuint shaderProgramID, Camera &camera);

	GLuint GenerateTexture(VolumeDataset &volume);
	void UpdateTexture(VolumeDataset &volume);
};

#endif

