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
	int epsilon;
	float extrapConst;
	int numBlocksCopied, numBlocksExtrapolated;

	int currentTimestep;

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

	TempCoherence(VolumeDataset &volume);
	GLuint GenerateTexture(VolumeDataset &volume);

	GLuint TemporalCoherence(VolumeDataset &volume, int currentTimestep_);
	void GPUPredict(VolumeDataset &volume);
	void CPUPredict(VolumeDataset &volume);
	bool BlockCompare(VolumeDataset &volume, int x, int y, int z);

	void CopyBlockToGPU(VolumeDataset &volume, cudaArray *nextArry, int x, int y, int z);
	void CopyBlockToChunk(VolumeDataset &volume, int x, int y, int z);
	void CopyChunkToGPU(VolumeDataset &volume);
};

#endif

