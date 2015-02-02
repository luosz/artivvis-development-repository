#include "TempCoherence.h"

texture <unsigned char, cudaTextureType3D, cudaReadModeElementType> prevTexRef;
texture <unsigned char, cudaTextureType3D, cudaReadModeElementType> currTexRef;
texture <unsigned char, cudaTextureType3D, cudaReadModeElementType> nextTexRef;

TempCoherence::TempCoherence(int screenWidth, int screenHeight, VolumeDataset &volume)
{
	blockRes = 8;
	alpha = 6;

	histogram = new FrequencyHistogram();
//	histogram = new VisibilityHistogram(screenWidth, screenHeight);

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

	threads.resize(NUM_THREADS);
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

void TempCoherence::MapTexturesToCuda()
{
	HANDLE_ERROR( cudaGraphicsGLRegisterImage(&cudaResources[0], prevTexture3D, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsNone) );
	HANDLE_ERROR( cudaGraphicsGLRegisterImage(&cudaResources[1], currTexture3D, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsNone) );
	HANDLE_ERROR( cudaGraphicsGLRegisterImage(&cudaResources[2], nextTexture3D, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore) );

	HANDLE_ERROR( cudaGraphicsMapResources(3, &cudaResources[0]) );

	prevArry = 0;	
	HANDLE_ERROR( cudaGraphicsSubResourceGetMappedArray(&prevArry, cudaResources[0], 0, 0) ); 
	HANDLE_ERROR( cudaBindTextureToArray(prevTexRef, prevArry) );

	currArry = 0;	
	HANDLE_ERROR( cudaGraphicsSubResourceGetMappedArray(&currArry, cudaResources[1], 0, 0) ); 
	HANDLE_ERROR( cudaBindTextureToArray(currTexRef, currArry) );

	nextArry = 0;	
	HANDLE_ERROR( cudaGraphicsSubResourceGetMappedArray(&nextArry, cudaResources[2], 0, 0) );
}


void TempCoherence::UnmapTextures()
{
	// Unbind and unmap, must be done before OpenGL uses texture memory again
	HANDLE_ERROR( cudaUnbindTexture(prevTexRef) );
	HANDLE_ERROR( cudaUnbindTexture(currTexRef) );
	HANDLE_ERROR( cudaUnbindTexture(nextTexRef) );

	HANDLE_ERROR( cudaGraphicsUnmapResources(3, &cudaResources[0]) );

	HANDLE_ERROR( cudaGraphicsUnregisterResource(cudaResources[0]) );
	HANDLE_ERROR( cudaGraphicsUnregisterResource(cudaResources[1]) );
	HANDLE_ERROR( cudaGraphicsUnregisterResource(cudaResources[2]) );
}

void TempCoherence::GPUPredict(VolumeDataset &volume)
{
	cudaResourceDesc wdsc;
	wdsc.resType = cudaResourceTypeArray;
	wdsc.res.array.array = nextArry;
	cudaSurfaceObject_t writeSurface;
	HANDLE_ERROR( cudaCreateSurfaceObject(&writeSurface, &wdsc) );

	CudaPredict <<<(volume.numVoxels + 255) / 256, 256>>>(volume.numVoxels, volume.xRes, volume.yRes, volume.zRes, writeSurface);
}


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

	for (int k=0; k<blockRes; k+=CHECK_STRIDE)
	{
		for (int j=0; j<blockRes; j+=CHECK_STRIDE)
		{
			for (int i=0; i<blockRes; i+=CHECK_STRIDE)
			{

				if ((xMin + i) >= volume.xRes || (yMin + j) >= volume.yRes || (zMin + k) >= volume.zRes)
					continue;

				ID = (xMin + i) + ((yMin + j) * volume.xRes) + ((zMin + k) * volume.xRes * volume.yRes);

				unsigned char p = nextTempVolume[ID];
				unsigned char n = nextVolume[ID];

//				if (n <= alpha)
//					beta = (float)n / float(alpha);
//				else
//					beta = ((float)(255 - n)) / ((float)(255 - alpha));

//				omega = (float)histogram->values[n];

				int diff =  n - p;
				
				top += histogram->values[n] * diff * diff;
			}
		}
	}

	int numPerAxis = blockRes / CHECK_STRIDE;
	bottom = numPerAxis * numPerAxis * numPerAxis;

	float similar = glm::sqrt(top / bottom);

	if (similar < EPSILON)
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



void TempCoherence::CopyBlockToGPU(VolumeDataset &volume, int x, int y, int z)
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


void TempCoherence::CopyBlockToChunk(VolumeDataset &volume, int posInChunk, int x, int y, int z)
{
	GLubyte *currentTimeAddress = volume.memblock3D + (currentTimestep * volume.numVoxels);
	cudaExtent extent = make_cudaExtent(blockRes, blockRes, blockRes);

	cudaMemcpy3DParms cudaCpyParams = {0};
	cudaCpyParams.kind = cudaMemcpyHostToHost;
	cudaCpyParams.extent = extent;

	cudaCpyParams.srcPos = make_cudaPos((x * blockRes), (y * blockRes), (z * blockRes));
	cudaCpyParams.srcPtr = make_cudaPitchedPtr((void*)currentTimeAddress, volume.xRes, volume.yRes, volume.zRes);

	cudaCpyParams.dstPos = make_cudaPos((posInChunk * blockRes), 0, 0);
	cudaCpyParams.dstPtr = make_cudaPitchedPtr((void*)chunkToBeCopied, numBlocks * blockRes, blockRes, blockRes);
	
	cudaMemcpy3D(&cudaCpyParams) ;
}


void TempCoherence::CPUExtrap(int begin, int end)
{
	for (int i=begin; i<end; i+=1)
	{
		int temp = (EXTRAP_CONST * currTempVolume[i]) - prevTempVolume[i];
		nextTempVolume[i] = (unsigned char)glm::clamp(temp, 0, 255);
	}
}


void TempCoherence::CPUCompare(int begin, int end, VolumeDataset &volume)
{
	for (int i=begin; i<end; i++)
	{
		int z = i / (numXBlocks * numYBlocks);
		int remainder = i % (numXBlocks * numYBlocks);

		int y = remainder / numXBlocks;

		int x = remainder % numXBlocks;

		if (BlockCompare(volume, x, y, z) == false)
		{
			CopyBlockToGPU(std::ref(volume), x, y, z);
//			int posInChunk = atomicNumBlocksCopied.fetch_add(1);
//			blocksToBeCopied[posInChunk] = BlockID(x, y, z);
//			CopyBlockToChunk(volume, posInChunk, x, y, z);
		}
	}
}

void TempCoherence::CPUPredict(VolumeDataset &volume)
{
	int beginID = 0;
	int numPerThread = volume.numVoxels / NUM_THREADS;

	for (int i=0; i<NUM_THREADS-1; i++)
	{
		threads[i] = std::thread(&TempCoherence::CPUExtrap, this, beginID, beginID + numPerThread);
		beginID += numPerThread;
	}
	threads[NUM_THREADS-1] = std::thread(&TempCoherence::CPUExtrap, this, beginID, volume.numVoxels);

	for (int i=0; i<NUM_THREADS; i++)
		threads[i].join();

	std::memcpy(&prevTempVolume[0], &currTempVolume[0], volume.numVoxels * volume.bytesPerElement);
	std::memcpy(&currTempVolume[0], &nextTempVolume[0], volume.numVoxels * volume.bytesPerElement);

	numPerThread = numBlocks / NUM_THREADS;
	beginID = 0;

	for (int i=0; i<NUM_THREADS-1; i++)
	{
		threads[i] = std::thread(&TempCoherence::CPUCompare, this, beginID, beginID + numPerThread, std::ref(volume));
		beginID += numPerThread;
	}
	threads[NUM_THREADS-1] = std::thread(&TempCoherence::CPUCompare, this, beginID, numBlocks, std::ref(volume));

	for (int i=0; i<NUM_THREADS; i++)
		threads[i].join();
	
	numBlocksCopied = atomicNumBlocksCopied.load();		
}

/*
void TempCoherence::CPUPredict(VolumeDataset &volume)
{
	// Beware of this, think it requires even stepsize
	for (int i=0; i<volume.numVoxels; i+=CHECK_STRIDE)
	{
		int temp = (EXTRAP_CONST * currTempVolume[i]) - prevTempVolume[i];
		nextTempVolume[i] = (unsigned char)glm::clamp(temp, 0, 255);

		prevTempVolume[i] = currTempVolume[i];
		currTempVolume[i] = nextTempVolume[i];	
	}

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
*/


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


GLuint TempCoherence::TemporalCoherence(VolumeDataset &volume, int currentTimestep_, TransferFunction &tf, ShaderManager &shaderManager, Camera &camera)
{
	currentTimestep = currentTimestep_;
	numBlocksCopied = numBlocksExtrapolated = 0;
	atomicNumBlocksCopied = 0;

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
		MapTexturesToCuda();
		GPUPredict(volume);
		histogram->Update(currentTimestep, volume, currTexture3D, tf.tfTexture, shaderManager, camera);
		CPUPredict(volume);
		UnmapTextures();
//		CopyChunkToGPU(volume);
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

	glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, volume.xRes, volume.yRes, volume.zRes, 0,  GL_RED, GL_UNSIGNED_BYTE, (volume.memblock3D + (textureSize * currentTimestep)));

	glBindTexture(GL_TEXTURE_3D, 0);

	return tex;
}




/*
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

				if (absDiff > EPSILON)
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
*/