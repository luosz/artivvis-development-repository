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


struct BlockID
{
	int x, y, z;

	BlockID() { }

	BlockID(int x_, int y_, int z_)
	{
		x = x_;
		y = y_;
		z = z_;
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
	
	std::vector<BlockID> blocksToBeCopied;
	unsigned char *chunkToBeCopied;
	unsigned char *cudaCopiedChunk;

	int alpha;
	int maxFrequency;
	int nonZeroFrequencies;
	std::vector<int> frequencyHistogram;

	float maxRatio, minRatio, meanRatio, stdDev;
	std::vector<float> ratios;
	int ratioTimeSteps;

	void TemporalCoherence(VolumeDataset &volume);
	void GPUPredict(VolumeDataset &volume);
	void CPUPredict(VolumeDataset &volume);
	bool BlockCompare(VolumeDataset &volume, int x, int y, int z);

	void CopyBlockToGPU(VolumeDataset &volume, cudaArray *nextArry, int x, int y, int z);
	void CopyBlockToChunk(VolumeDataset &volume, int x, int y, int z);
	void CopyChunkToGPU(VolumeDataset &volume);

	BlockRaycaster(int screenWidth, int screenHeight, VolumeDataset &volume);
	void Raycast(VolumeDataset &volume, TransferFunction &transferFunction, GLuint shaderProgramID, Camera &camera);
	
	void BlockRaycast(VolumeDataset &volume, TransferFunction &transferFunction, GLuint shaderProgramID, Camera &camera);

	GLuint GenerateTexture(VolumeDataset &volume);
};

#endif

