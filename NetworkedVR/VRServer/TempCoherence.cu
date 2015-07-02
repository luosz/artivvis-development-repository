#include "TempCoherence.h"

texture <unsigned char, cudaTextureType3D, cudaReadModeElementType> prevTexRef;
texture <unsigned char, cudaTextureType3D, cudaReadModeElementType> currTexRef;
texture <unsigned char, cudaTextureType3D, cudaReadModeElementType> nextTexRef;
texture <unsigned char, cudaTextureType3D, cudaReadModeElementType> exactTexRef;



TempCoherence::TempCoherence(int screenWidth, int screenHeight, VolumeDataset &volume, NetworkManager *networkManager)
{
	blockRes = BLOCK_RES;
	alpha = 6;

	netManager = networkManager;

	histogram = new FrequencyHistogram();

	numXBlocks = glm::ceil((float)volume.xRes / (float)blockRes);
	numYBlocks = glm::ceil((float)volume.yRes / (float)blockRes);
	numZBlocks = glm::ceil((float)volume.zRes / (float)blockRes);
	numBlocks = numXBlocks * numYBlocks * numZBlocks;

	textureSize = volume.xRes * volume.yRes * volume.zRes * volume.bytesPerElement;
	prevTexture3D = volume.GenerateTexture();
	currTexture3D = volume.GenerateTexture();
	nextTexture3D = volume.GenerateTexture();
	exactTexture3D = volume.GenerateTexture();

	for (int i=0; i<4; i++)
		cudaResources.push_back(cudaGraphicsResource_t());

	HANDLE_ERROR( cudaMalloc((void**)&cudaBlockFlags, numBlocks * sizeof(bool)) );
	hostBlockFlags = new bool[numBlocks];
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

__global__ void CudaBlockCompare(int numBlocks, int blockRes, int numXBlocks, int numYBlocks, int numZBlocks, int volumeXRes, int volumeYRes, int volumeZRes, bool *cudaBlockFlags, cudaSurfaceObject_t surface)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < numBlocks)
	{
		int z = tid / (numXBlocks * numYBlocks);
		int remainder = tid % (numXBlocks * numYBlocks);

		int y = remainder / numXBlocks;

		int x = remainder % numXBlocks;

		int xMin = x * blockRes;
		int yMin = y * blockRes;
		int zMin = z * blockRes;

		float top, bottom;
		top = bottom = 0.0f;

		for (int k=0; k<blockRes; k+=CHECK_STRIDE)
		{
			for (int j=0; j<blockRes; j+=CHECK_STRIDE)
			{
				for (int i=0; i<blockRes; i+=CHECK_STRIDE)
				{
					if ((xMin + i) >= volumeXRes || (yMin + j) >= volumeYRes || (zMin + k) >= volumeZRes)
						continue;

					unsigned char predictVal, exactVal;
				
					surf3Dread(&predictVal, surface, xMin + i, yMin + j, zMin + k);
					exactVal = tex3D(exactTexRef, xMin + i, yMin + j, zMin + k);

					int diff =  exactVal - predictVal;
					
					top += diff * diff;
				}
			}
		}

		int numPerAxis = blockRes / CHECK_STRIDE;
		bottom = numPerAxis * numPerAxis * numPerAxis;

		float similar = glm::sqrt(top / bottom);

		if (similar > EPSILON)
		{
			cudaBlockFlags[tid] = true;

			for (int k=0; k<blockRes; k+=CHECK_STRIDE)
				for (int j=0; j<blockRes; j+=CHECK_STRIDE)
					for (int i=0; i<blockRes; i+=CHECK_STRIDE)
					{
						if ((xMin + i) >= volumeXRes || (yMin + j) >= volumeYRes || (zMin + k) >= volumeZRes)
							continue;

						unsigned char exactVal = tex3D(exactTexRef, xMin + i, yMin + j, zMin + k);

						surf3Dwrite(exactVal, surface, xMin + i, yMin + j, zMin + k);
					}
		}
	}
}

void TempCoherence::MapTexturesToCuda()
{
	HANDLE_ERROR( cudaGraphicsGLRegisterImage(&cudaResources[0], prevTexture3D, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsNone) );
	HANDLE_ERROR( cudaGraphicsGLRegisterImage(&cudaResources[1], currTexture3D, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsNone) );
	HANDLE_ERROR( cudaGraphicsGLRegisterImage(&cudaResources[2], nextTexture3D, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore) );
	HANDLE_ERROR( cudaGraphicsGLRegisterImage(&cudaResources[3], exactTexture3D, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsNone) );

	HANDLE_ERROR( cudaGraphicsMapResources(4, &cudaResources[0]) );

	prevArry = 0;	
	HANDLE_ERROR( cudaGraphicsSubResourceGetMappedArray(&prevArry, cudaResources[0], 0, 0) ); 
	HANDLE_ERROR( cudaBindTextureToArray(prevTexRef, prevArry) );

	currArry = 0;	
	HANDLE_ERROR( cudaGraphicsSubResourceGetMappedArray(&currArry, cudaResources[1], 0, 0) ); 
	HANDLE_ERROR( cudaBindTextureToArray(currTexRef, currArry) );

	nextArry = 0;	
	HANDLE_ERROR( cudaGraphicsSubResourceGetMappedArray(&nextArry, cudaResources[2], 0, 0) );

	exactArry = 0;	
	HANDLE_ERROR( cudaGraphicsSubResourceGetMappedArray(&exactArry, cudaResources[3], 0, 0) ); 
	HANDLE_ERROR( cudaBindTextureToArray(exactTexRef, exactArry) );
}


void TempCoherence::UnmapTextures()
{
	// Unbind and unmap, must be done before OpenGL uses texture memory again
	HANDLE_ERROR( cudaUnbindTexture(prevTexRef) );
	HANDLE_ERROR( cudaUnbindTexture(currTexRef) );
	HANDLE_ERROR( cudaUnbindTexture(nextTexRef) );
	HANDLE_ERROR( cudaUnbindTexture(exactTexRef) );

	HANDLE_ERROR( cudaGraphicsUnmapResources(4, &cudaResources[0]) );

	HANDLE_ERROR( cudaGraphicsUnregisterResource(cudaResources[0]) );
	HANDLE_ERROR( cudaGraphicsUnregisterResource(cudaResources[1]) );
	HANDLE_ERROR( cudaGraphicsUnregisterResource(cudaResources[2]) );
	HANDLE_ERROR( cudaGraphicsUnregisterResource(cudaResources[3]) );
}

void TempCoherence::GPUPredict(VolumeDataset &volume)
{
	cudaResourceDesc wdsc;
	wdsc.resType = cudaResourceTypeArray;
	wdsc.res.array.array = nextArry;
	cudaSurfaceObject_t writeSurface;
	HANDLE_ERROR( cudaCreateSurfaceObject(&writeSurface, &wdsc) );

	CudaPredict <<<(volume.numVoxels + 255) / 256, 256>>>(volume.numVoxels, volume.xRes, volume.yRes, volume.zRes, writeSurface);

	HANDLE_ERROR( cudaDeviceSynchronize() );

	CudaBlockCompare <<<(numBlocks + 255) / 256, 256>>>(numBlocks, blockRes, numXBlocks, numYBlocks, numZBlocks, volume.xRes, volume.yRes, volume.zRes, cudaBlockFlags, writeSurface);

	HANDLE_ERROR( cudaDeviceSynchronize() );
}



GLuint TempCoherence::TemporalCoherence(VolumeDataset &volume, int currentTimestep)
{
	numBlocksCopied = numBlocksExtrapolated = 0;
	atomicNumBlocksCopied = 0;

	GLuint temp = prevTexture3D;
	prevTexture3D = currTexture3D;
	currTexture3D = nextTexture3D;
	nextTexture3D = temp;

	HANDLE_ERROR( cudaMemset(cudaBlockFlags, 0, numBlocks * sizeof(bool)) );

	if (currentTimestep < 2)
	{
		glBindTexture(GL_TEXTURE_3D, nextTexture3D);
		glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, volume.xRes, volume.yRes, volume.zRes, GL_RED, GL_UNSIGNED_BYTE, volume.currMemblock);
		glBindTexture(GL_TEXTURE_3D, 0);
		
		netManager->SendState(numXBlocks, numYBlocks, numZBlocks, blockRes);
	}
	else
	{
		glBindTexture(GL_TEXTURE_3D, exactTexture3D);
		glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, volume.xRes, volume.yRes, volume.zRes, GL_RED, GL_UNSIGNED_BYTE, volume.currMemblock);
		glBindTexture(GL_TEXTURE_3D, 0);

		MapTexturesToCuda();
		GPUPredict(volume);
//		histogram->Update(currentTimestep, volume, currTexture3D);
		UnmapTextures();

		HANDLE_ERROR( cudaMemcpy(hostBlockFlags, cudaBlockFlags, numBlocks * sizeof(bool), cudaMemcpyDeviceToHost) );

		int blockID = 0;
		for (int k=0; k<numZBlocks; k++)
			for (int j=0; j<numYBlocks; j++)
				for (int i=0; i<numXBlocks; i++)
				{
					if (hostBlockFlags[blockID])
					{
						numBlocksCopied++;
						netManager->SendBlock(i, j, k, blockRes);
					}
					else
						numBlocksExtrapolated++;

					blockID++;
				}


		std::cout << "copied: " << numBlocksCopied << "   -   extrapolated: " << numBlocksExtrapolated << std::endl;
	}
	
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

	glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, volume.xRes, volume.yRes, volume.zRes, 0,  GL_RED, GL_UNSIGNED_BYTE, volume.currMemblock);

	glBindTexture(GL_TEXTURE_3D, 0);

	return tex;


}

