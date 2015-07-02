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
#include "FrequencyHistogram.h"
#include <thread>
#include <atomic>
#include "ServerNetworkManager.h"

#define EXTRAP_CONST 2
#define EPSILON 0.25f
#define CHECK_STRIDE 1
#define NUM_THREADS 7
#define BLOCK_RES 4


class TempCoherence
{
public:
	NetworkManager *netManager;

	int blockRes;
	int numBlocks;
	int numXBlocks, numYBlocks, numZBlocks;

	int textureSize;
	GLuint currTexture3D;
	GLuint prevTexture3D;
	GLuint nextTexture3D;
	GLuint exactTexture3D;
	
	std::atomic<int> atomicNumBlocksCopied;
	int numBlocksCopied, numBlocksExtrapolated;

	std::vector<cudaGraphicsResource_t> cudaResources;
	cudaArray *prevArry, *currArry, *nextArry, *exactArry;

	bool *cudaBlockFlags;
	bool *hostBlockFlags;

	int alpha;
	FrequencyHistogram *histogram;

	TempCoherence(int screenWidth, int screenHeight, VolumeDataset &volume, NetworkManager *networkManager);
	GLuint GenerateTexture(VolumeDataset &volume);

	GLuint TemporalCoherence(VolumeDataset &volume, int currentTimestep_);
	void GPUPredict(VolumeDataset &volume);
	void MapTexturesToCuda();
	void UnmapTextures();
};

#endif

