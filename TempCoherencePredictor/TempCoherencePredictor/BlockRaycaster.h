#ifndef BLOCK_RAYCASTER_H
#define BLOCK_RAYCASTER_H

#include <GL/glew.h>
#include <GL/freeglut.h>
#include "Camera.h"
#include "GLM.h"
#include <vector>
#include "VolumeDataset.h"
#include "TransferFunction.h"


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
	int numBlocks;

	BlockRaycaster(int screenWidth, int screenHeight, VolumeDataset &volume);
	void Raycast(VolumeDataset &volume, TransferFunction &transferFunction, GLuint shaderProgramID, Camera &camera);
};

#endif

