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

#define EXTRAP_CONST 2
#define EPSILON 0.5f
#define CHECK_STRIDE 1
#define NUM_THREADS 7

class TempCoherence
{
public:
	int blockRes;
	int numBlocks;
	int numXBlocks, numYBlocks, numZBlocks;

	TempCoherence(int screenWidth, int screenHeight, VolumeDataset &volume);
};

#endif

