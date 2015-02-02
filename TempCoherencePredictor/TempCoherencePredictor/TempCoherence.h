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
#include "VisibilityHistogram.h"
#include <thread>
#include <atomic>

#define EXTRAP_CONST 2
#define EPSILON 1.0f
#define CHECK_STRIDE 1
#define NUM_THREADS 7


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

class TempCoherence
{
public:
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
	
	std::atomic<int> atomicNumBlocksCopied;
	int numBlocksCopied, numBlocksExtrapolated;
	int currentTimestep;

	std::vector<cudaGraphicsResource_t> cudaResources;
	cudaArray *prevArry, *currArry, *nextArry;
	
	std::vector<BlockID> blocksToBeCopied;
	unsigned char *chunkToBeCopied;
	unsigned char *cudaCopiedChunk;

	int alpha;
	Histogram *histogram;

	std::vector<std::thread> threads;

	TempCoherence(int screenWidth, int screenHeight, VolumeDataset &volume);
	GLuint GenerateTexture(VolumeDataset &volume);

	GLuint TemporalCoherence(VolumeDataset &volume, int currentTimestep_, TransferFunction &tf, ShaderManager &shaderManager, Camera &camera);
	void GPUPredict(VolumeDataset &volume);
	void CPUPredict(VolumeDataset &volume);
	bool BlockCompare(VolumeDataset &volume, int x, int y, int z);
	void CPUExtrap(int begin, int end);
	void CPUCompare(int begin, int end, VolumeDataset &volume);

	void MapTexturesToCuda();
	void UnmapTextures();

	void CopyBlockToGPU(VolumeDataset &volume, int x, int y, int z);
	void CopyBlockToChunk(VolumeDataset &volume, int x, int y, int z);
	void CopyBlockToChunk(VolumeDataset &volume, int posInChunk, int x, int y, int z);
	void CopyChunkToGPU(VolumeDataset &volume);
};

#endif

