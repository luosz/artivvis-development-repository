#include "TempCoherence.h"

texture <unsigned char, cudaTextureType3D, cudaReadModeElementType> prevTexRef;
texture <unsigned char, cudaTextureType3D, cudaReadModeElementType> currTexRef;
texture <unsigned char, cudaTextureType3D, cudaReadModeElementType> nextTexRef;

TempCoherence::TempCoherence(VolumeDataset &volume)
{
	epsilon = 10;
	blockRes = 8;
	alpha = 6;

	numXBlocks = glm::ceil((float)volume.xRes / (float)blockRes);
	numYBlocks = glm::ceil((float)volume.yRes / (float)blockRes);
	numZBlocks = glm::ceil((float)volume.zRes / (float)blockRes);
	numBlocks = numXBlocks * numYBlocks * numZBlocks;

	float xVoxelWidth = 2.0f / (float) volume.xRes;
	float yVoxelWidth = 2.0f / (float) volume.yRes;
	float zVoxelWidth = 2.0f / (float) volume.zRes;

	currentTimestep = 0;

	textureSize = volume.xRes * volume.yRes * volume.zRes * volume.bytesPerElement;
	prevTexture3D = GenerateTexture(volume);
	currTexture3D = GenerateTexture(volume);
	nextTexture3D = GenerateTexture(volume);

	for (int i=0; i<3; i++)
		cudaResources.push_back(cudaGraphicsResource_t());

	prevTempVolume = new unsigned char[volume.numVoxels * volume.bytesPerElement];
	currTempVolume = new unsigned char[volume.numVoxels * volume.bytesPerElement];
	nextTempVolume = new unsigned char[volume.numVoxels * volume.bytesPerElement];
	chunkToBeCopied = new unsigned char[numBlocks * blockRes * blockRes * blockRes];

	HANDLE_ERROR( cudaMalloc((void**)&cudaCopiedChunk, numBlocks * blockRes * blockRes * blockRes) );

	blocksToBeCopied.resize(numBlocks);

	frequencyHistogram.resize(256);

	ratioTimeSteps = 200;
	ratios.resize(ratioTimeSteps);
	std::fill(ratios.begin(), ratios.end(), 0.0f);
}

__global__ void CudaPredict(int numVoxels, int xRes, int yRes, int zRes, cudaSurfaceObject_t surface)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < numVoxels)
	{
		int z = tid / (xRes * yRes);
		int remainder = tid % (xRes * yRes);

		int y = remainder / xRes;

		int x = remainder % xRes;

		unsigned char prevVal, currVal, nextVal;

		prevVal = tex3D(prevTexRef, x, y, z);
		currVal = tex3D(currTexRef, x, y, z);

		int temp = (EXTRAP_CONST * currVal) - prevVal;
		nextVal = (unsigned char)glm::clamp(temp, 0, 255);

		surf3Dwrite(nextVal, surface, x, y, z);
	}
}

void TempCoherence::GPUPredict(VolumeDataset &volume)
{
	HANDLE_ERROR( cudaGraphicsGLRegisterImage(&cudaResources[0], prevTexture3D, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsNone) );
	HANDLE_ERROR( cudaGraphicsGLRegisterImage(&cudaResources[1], currTexture3D, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsNone) );
	HANDLE_ERROR( cudaGraphicsGLRegisterImage(&cudaResources[2], nextTexture3D, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore) );

	HANDLE_ERROR( cudaGraphicsMapResources(3, &cudaResources[0]) );

	cudaArray *prevArry = 0;	
	HANDLE_ERROR( cudaGraphicsSubResourceGetMappedArray(&prevArry, cudaResources[0], 0, 0) ); 
	HANDLE_ERROR( cudaBindTextureToArray(prevTexRef, prevArry) );

	cudaArray *currArry = 0;	
	HANDLE_ERROR( cudaGraphicsSubResourceGetMappedArray(&currArry, cudaResources[1], 0, 0) ); 
	HANDLE_ERROR( cudaBindTextureToArray(currTexRef, currArry) );

	cudaArray *nextArry = 0;	
	HANDLE_ERROR( cudaGraphicsSubResourceGetMappedArray(&nextArry, cudaResources[2], 0, 0) ); 

	cudaResourceDesc wdsc;
	wdsc.resType = cudaResourceTypeArray;
	wdsc.res.array.array = nextArry;
	cudaSurfaceObject_t writeSurface;
	HANDLE_ERROR( cudaCreateSurfaceObject(&writeSurface, &wdsc) );

	CudaPredict <<<(volume.numVoxels + 255) / 256, 256>>>(volume.numVoxels, volume.xRes, volume.yRes, volume.zRes, writeSurface);

	// Unbind and unmap, must be done before OpenGL uses texture memory again
	HANDLE_ERROR( cudaUnbindTexture(prevTexRef) );
	HANDLE_ERROR( cudaUnbindTexture(currTexRef) );
	HANDLE_ERROR( cudaUnbindTexture(nextTexRef) );

	HANDLE_ERROR( cudaGraphicsUnmapResources(3, &cudaResources[0]) );

	HANDLE_ERROR( cudaGraphicsUnregisterResource(cudaResources[0]) );
	HANDLE_ERROR( cudaGraphicsUnregisterResource(cudaResources[1]) );
	HANDLE_ERROR( cudaGraphicsUnregisterResource(cudaResources[2]) );

}

/*
bool TempCoherence::BlockCompare(VolumeDataset &volume, int x, int y, int z)
{
	GLubyte *nextVolume = volume.memblock3D + (currentTimestep * volume.numVoxels);

	int xMin = x * blockRes;
	int yMin = y * blockRes;
	int zMin = z * blockRes;

	int ID;
	float omega, beta;
	float top, bottom;
	top = bottom = 0.0f;

	for (int k=0; k<blockRes; k++)
		for (int j=0; j<blockRes; j++)
			for (int i=0; i<blockRes; i++)
			{
				if ((xMin + i) >= volume.xRes || (yMin + j) >= volume.yRes || (zMin + k) >= volume.zRes)
					continue;

				ID = (xMin + i) + ((yMin + j) * volume.xRes) + ((zMin + k) * volume.xRes * volume.yRes);

				unsigned char p = nextTempVolume[ID];
				unsigned char n = nextVolume[ID];

				if (n <= alpha)
					beta = (float)n / float(alpha);
				else
					beta = ((float)(255 - n)) / ((float)(255 - alpha));

				omega = ((float)frequencyHistogram[n] / (float) maxFrequency);

//				omega = beta;

				int diff =  n - p;
				
				top += omega * glm::pow(diff, 2);

//				bottom += omega;
			}

//	bottom *= nonZeroFrequencies;
	bottom = blockRes * blockRes * blockRes;

	float similar = glm::sqrt(top / bottom);
//	similar = glm::sqrt(top);

	if (similar < (float)epsilon)
		return true;


	for (int k=0; k<blockRes; k++)
		for (int j=0; j<blockRes; j++)
			for (int i=0; i<blockRes; i++)
			{
				if ((xMin + i) >= volume.xRes || (yMin + j) >= volume.yRes || (zMin + k) >= volume.zRes)
					continue;

				ID = (xMin + i) + ((yMin + j) * volume.xRes) + ((zMin + k) * volume.xRes * volume.yRes);

				currTempVolume[ID] = nextVolume[ID];
			}

	return false;
}
*/


void TempCoherence::CopyBlockToGPU(VolumeDataset &volume, cudaArray *nextArry, int x, int y, int z)
{
	GLubyte *currentTimeAddress = volume.memblock3D + (currentTimestep * volume.numVoxels);
	cudaPos offset = make_cudaPos((x * blockRes), (y * blockRes), (z * blockRes));
	cudaExtent extent = make_cudaExtent(blockRes, blockRes, blockRes);

	cudaMemcpy3DParms cudaCpyParams = {0};
	cudaCpyParams.kind = cudaMemcpyHostToDevice;
	cudaCpyParams.extent = extent;

	cudaCpyParams.dstPos = offset;
	cudaCpyParams.dstArray = nextArry;
	
	cudaCpyParams.srcPos = offset;
	cudaCpyParams.srcPtr = make_cudaPitchedPtr((void*)currentTimeAddress, volume.xRes, volume.yRes, volume.zRes);

	cudaMemcpy3D(&cudaCpyParams);
}

void TempCoherence::CopyBlockToChunk(VolumeDataset &volume, int x, int y, int z)
{
	GLubyte *currentTimeAddress = volume.memblock3D + (currentTimestep * volume.numVoxels);
	cudaExtent extent = make_cudaExtent(blockRes, blockRes, blockRes);

//	if (x == numXBlocks - 1)
//		extent.width = volume.xRes % blockRes;
//	if (y == numYBlocks - 1)
//		extent.height = volume.yRes % blockRes;
//	if (z == numZBlocks - 1)
//		extent.depth = volume.zRes % blockRes;

	cudaMemcpy3DParms cudaCpyParams = {0};
	cudaCpyParams.kind = cudaMemcpyHostToHost;
	cudaCpyParams.extent = extent;

	cudaCpyParams.srcPos = make_cudaPos((x * blockRes), (y * blockRes), (z * blockRes));
	cudaCpyParams.srcPtr = make_cudaPitchedPtr((void*)currentTimeAddress, volume.xRes, volume.yRes, volume.zRes);

	cudaCpyParams.dstPos = make_cudaPos((numBlocksCopied * blockRes), 0, 0);
	cudaCpyParams.dstPtr = make_cudaPitchedPtr((void*)chunkToBeCopied, numBlocks * blockRes, blockRes, blockRes);
	
	cudaMemcpy3D(&cudaCpyParams) ;
}

void TempCoherence::CPUPredict(VolumeDataset &volume)
{
	std::fill(frequencyHistogram.begin(), frequencyHistogram.end(), 0);

	// Beware of this, think it requires even stepsize
	for (int i=0; i<volume.numVoxels; i++)
	{
		int temp = (EXTRAP_CONST * currTempVolume[i]) - prevTempVolume[i];
		nextTempVolume[i] = (unsigned char)glm::clamp(temp, 0, 255);

		prevTempVolume[i] = currTempVolume[i];
		currTempVolume[i] = nextTempVolume[i];	
	}

	for (int i=0; i<volume.numVoxels; i++)
	{
		int bucket = volume.memblock3D[(currentTimestep*volume.numVoxels) + i];
		frequencyHistogram[bucket]++;
	}

	maxFrequency = nonZeroFrequencies = 0;
	for (int i=1; i<256; i++)
	{
		int freq = frequencyHistogram[i];
		maxFrequency = glm::max(maxFrequency, freq);
		nonZeroFrequencies += freq;
	}
	frequencyHistogram[0] = maxFrequency;

	for (int z=0; z<numZBlocks; z++)
		for (int y =0; y<numYBlocks; y++)
			for (int x=0; x<numXBlocks; x++)
			{
				if (BlockCompare(volume, x, y, z) == false)
				{
					blocksToBeCopied[numBlocksCopied] = BlockID(x, y, z);
					CopyBlockToChunk(volume, x, y, z);

					numBlocksCopied++;
				}
				else
					numBlocksExtrapolated++;
			} 
}


void TempCoherence::CopyChunkToGPU(VolumeDataset &volume)
{
	cudaExtent extent = make_cudaExtent(numBlocksCopied * blockRes, blockRes, blockRes);

	cudaMemcpy3DParms cudaCpyParams = {0};
	cudaCpyParams.kind = cudaMemcpyHostToDevice;
	cudaCpyParams.extent = extent;

	cudaCpyParams.srcPtr = make_cudaPitchedPtr((void*)chunkToBeCopied, numBlocks * blockRes, blockRes, blockRes);

	cudaCpyParams.dstPtr = make_cudaPitchedPtr((void*)cudaCopiedChunk, numBlocks * blockRes, blockRes, blockRes);
	
	cudaMemcpy3D(&cudaCpyParams);


	HANDLE_ERROR( cudaGraphicsGLRegisterImage(&cudaResources[0], nextTexture3D, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsNone) );
	HANDLE_ERROR( cudaGraphicsMapResources(1, &cudaResources[0]) );
	cudaArray *nextArry = 0;	
	HANDLE_ERROR( cudaGraphicsSubResourceGetMappedArray(&nextArry, cudaResources[0], 0, 0) ); 
	HANDLE_ERROR( cudaBindTextureToArray(nextTexRef, nextArry) );


	extent = make_cudaExtent(blockRes, blockRes, blockRes);

	cudaCpyParams = cudaMemcpy3DParms();
	cudaCpyParams.kind = cudaMemcpyDeviceToDevice;
	cudaCpyParams.extent = extent;

	for (int i=0; i<numBlocksCopied; i++)
	{
		cudaCpyParams.srcPos = make_cudaPos((i * blockRes), 0, 0);
		cudaCpyParams.srcPtr = make_cudaPitchedPtr((void*)cudaCopiedChunk, numBlocks * blockRes, blockRes, blockRes);
		
		cudaCpyParams.dstPos = make_cudaPos((blocksToBeCopied[i].x * blockRes), (blocksToBeCopied[i].y * blockRes), (blocksToBeCopied[i].z * blockRes));
		cudaCpyParams.dstArray = nextArry;

		cudaMemcpy3D(&cudaCpyParams) ;
	}

	HANDLE_ERROR( cudaUnbindTexture(nextTexRef) );
	HANDLE_ERROR( cudaGraphicsUnmapResources(1, &cudaResources[0]) );
	HANDLE_ERROR( cudaGraphicsUnregisterResource(cudaResources[0]) );
}


GLuint TempCoherence::TemporalCoherence(VolumeDataset &volume, int currentTimestep_)
{
	currentTimestep = currentTimestep_;
	numBlocksCopied = numBlocksExtrapolated = 0;

	GLuint temp = prevTexture3D;
	prevTexture3D = currTexture3D;
	currTexture3D = nextTexture3D;
	nextTexture3D = temp;

	if (currentTimestep < 2)
	{
		glBindTexture(GL_TEXTURE_3D, nextTexture3D);
		glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, volume.xRes, volume.yRes, volume.zRes, 0,  GL_RED, GL_UNSIGNED_BYTE, (volume.memblock3D + (textureSize * currentTimestep)));
		glBindTexture(GL_TEXTURE_3D, 0);

		if (currentTimestep == 1)
		{
			for (int i=0; i<volume.numVoxels; i++)
			{
				prevTempVolume[i] = volume.memblock3D[i];
				currTempVolume[i] = volume.memblock3D[textureSize + i];
			}						
		}
	}
	else
	{
//		if (currentTimestep == ratioTimeSteps)
//		{
//			maxRatio = 0.0f;
//			minRatio = 100.0f;
//			meanRatio = 0.0f;
//			stdDev = 0.0f;
//
//			for (int i=2; i<ratioTimeSteps; i++)
//			{
//				maxRatio = glm::max(maxRatio, ratios[i]);
//				minRatio = glm::min(minRatio, ratios[i]);
//				meanRatio += ratios[i];
//			}
//
//			meanRatio /= ratioTimeSteps;
//
//			for (int i=2; i<ratioTimeSteps; i++)
//			{
//				stdDev += glm::pow((ratios[i] - meanRatio), 2.0f);
//			}
//
//			stdDev /= ratioTimeSteps;
//			stdDev = glm::sqrt(stdDev);
//
//			std::cout << "Max: " << maxRatio << std::endl;
//			std::cout << "Min: " << minRatio << std::endl;
//			std::cout << "Mean: " << meanRatio << std::endl;
//			std::cout << "StdDev: " << stdDev << std::endl;
////			getchar();
//		}

		GPUPredict(volume);
		CPUPredict(volume);

		CopyChunkToGPU(volume);


//		glBindTexture(GL_TEXTURE_3D, nextTexture3D);
//		glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, volume.xRes, volume.yRes, volume.zRes, 0,  GL_RED, GL_UNSIGNED_BYTE, (volume.memblock3D + (textureSize * currentTimestep)));
//		glBindTexture(GL_TEXTURE_3D, 0);

	}

//	std::cout << "Copied: " << numBlocksCopied << " - Extrapolated: " << numBlocksExtrapolated << std::endl;
//	ratios[currentTimestep] = (float)numBlocksExtrapolated / (float) numBlocks;
	
	return nextTexture3D;
}


GLuint TempCoherence::GenerateTexture(VolumeDataset &volume)
{
	GLuint tex;

	glEnable(GL_TEXTURE_3D);
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_3D, tex);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, volume.xRes, volume.yRes, volume.zRes, 0,  GL_RED, GL_UNSIGNED_BYTE, (volume.memblock3D + (textureSize * currentTimestep)));

	glBindTexture(GL_TEXTURE_3D, 0);

	return tex;
}





bool TempCoherence::BlockCompare(VolumeDataset &volume, int x, int y, int z)
{
	GLubyte *nextVolume = volume.memblock3D + (currentTimestep * volume.numVoxels);

	int xMin = x * blockRes;
	int yMin = y * blockRes;
	int zMin = z * blockRes;

	int ID;

	for (int k=0; k<blockRes; k++)
		for (int j=0; j<blockRes; j++)
			for (int i=0; i<blockRes; i++)
			{
				if ((xMin + i) >= volume.xRes || (yMin + j) >= volume.yRes || (zMin + k) >= volume.zRes)
					continue;

				ID = (xMin + i) + ((yMin + j) * volume.xRes) + ((zMin + k) * volume.xRes * volume.yRes);

				unsigned char p = nextTempVolume[ID];
				unsigned char n = nextVolume[ID];

				int diff =  p - n;
				int absDiff = glm::abs(diff);

				if (absDiff > epsilon)
					goto copy;
			}

	return true;

	// Put a goto to avoid an extra if, only gets here if entire block needs to be copied
	copy:
	for (int k=0; k<blockRes; k++)
		for (int j=0; j<blockRes; j++)
			for (int i=0; i<blockRes; i++)
			{
				if ((xMin + i) >= volume.xRes || (yMin + j) >= volume.yRes || (zMin + k) >= volume.zRes)
					continue;

				ID = (xMin + i) + ((yMin + j) * volume.xRes) + ((zMin + k) * volume.xRes * volume.yRes);

				currTempVolume[ID] = nextVolume[ID];
			}

	return false;
}
