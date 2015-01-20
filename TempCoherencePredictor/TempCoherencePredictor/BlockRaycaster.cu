#include "BlockRaycaster.h"

texture <unsigned char, cudaTextureType3D, cudaReadModeElementType> prevTexRef;
texture <unsigned char, cudaTextureType3D, cudaReadModeElementType> currTexRef;
texture <unsigned char, cudaTextureType3D, cudaReadModeElementType> nextTexRef;

BlockRaycaster::BlockRaycaster(int screenWidth, int screenHeight, VolumeDataset &volume)
{
	maxRaySteps = 1000;
	rayStepSize = 0.005f;
	gradientStepSize = 0.005f;

	lightPosition = glm::vec3(-0.0f, -5.0f, 5.0f);

	epsilon = 3;
	blockRes = 8;
	alpha = 6;

	numXBlocks = glm::ceil((float)volume.xRes / (float)blockRes);
	numYBlocks = glm::ceil((float)volume.yRes / (float)blockRes);
	numZBlocks = glm::ceil((float)volume.zRes / (float)blockRes);
	numBlocks = numXBlocks * numYBlocks * numZBlocks;

	float xVoxelWidth = 2.0f / (float) volume.xRes;
	float yVoxelWidth = 2.0f / (float) volume.yRes;
	float zVoxelWidth = 2.0f / (float) volume.zRes;

	blocks.reserve(numBlocks);

	for (int k=0; k<numZBlocks; k++)
		for (int j=0; j<numYBlocks; j++)
			for (int i=0; i<numXBlocks; i++)
			{
				blocks.push_back(Block(blockRes, i, j, k, xVoxelWidth, yVoxelWidth, zVoxelWidth));
			}

	currentTimestep = 0;
	oldTime = clock();

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

void BlockRaycaster::GPUPredict(VolumeDataset &volume)
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


bool BlockRaycaster::BlockCompare(VolumeDataset &volume, int x, int y, int z)
{
	GLubyte *nextVolume = volume.memblock3D + (currentTimestep * volume.numVoxels);

	int xMin = x * blockRes;
	int yMin = y * blockRes;
	int zMin = z * blockRes;

	int ID;
	float omega, beta;
	float top, bottom;
	top = bottom = 0.0f;

	for (int k=0; k<blockRes; k+=2)
		for (int j=0; j<blockRes; j+=2)
			for (int i=0; i<blockRes; i+=2)
			{
				if ((xMin + i) >= volume.xRes || (yMin + j) >= volume.yRes || (zMin + k) >= volume.zRes)
					continue;

				ID = (xMin + i) + ((yMin + j) * volume.xRes) + ((zMin + k) * volume.xRes * volume.yRes);

				unsigned char p = nextTempVolume[ID];
				unsigned char n = nextVolume[ID];

//				if (n <= alpha)
//					beta = (((float)n / float(alpha)) / 2.0f) + 0.5f;
//				else
//					beta = ((((float)(255 - n)) / ((float)(255 - alpha))) / 2.0f) + 0.5f;

				if (n <= alpha)
					beta = (float)n / float(alpha);
				else
					beta = ((float)(255 - n)) / ((float)(255 - alpha));

				omega = ((float)frequencyHistogram[n] / (float) maxFrequency);

//				omega = beta;

				int diff =  n - p;
				
				top += glm::pow(diff, 2);

				bottom += omega;
			}

//	bottom *= nonZeroFrequencies;
	bottom = blockRes * blockRes * blockRes;

	float similar = glm::sqrt(top / bottom);
//	similar = glm::sqrt(top);

	if (similar < (float)0.05f || bottom == 0)
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



void BlockRaycaster::CopyBlockToGPU(VolumeDataset &volume, cudaArray *nextArry, int x, int y, int z)
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

void BlockRaycaster::CopyBlockToChunk(VolumeDataset &volume, int x, int y, int z)
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

void BlockRaycaster::CPUPredict(VolumeDataset &volume)
{
	std::fill(frequencyHistogram.begin(), frequencyHistogram.end(), 0);

	// Beware of this, think it requires even stepsize
	for (int i=0; i<volume.numVoxels; i+=2)
	{
		int temp = (EXTRAP_CONST * currTempVolume[i]) - prevTempVolume[i];
		nextTempVolume[i] = (unsigned char)glm::clamp(temp, 0, 255);

		prevTempVolume[i] = currTempVolume[i];
		currTempVolume[i] = nextTempVolume[i];

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


void BlockRaycaster::CopyChunkToGPU(VolumeDataset &volume)
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


void BlockRaycaster::TemporalCoherence(VolumeDataset &volume)
{
	if (volume.timesteps > 1)
	{
		clock_t currentTime = clock();
		float time = (currentTime - oldTime) / (float) CLOCKS_PER_SEC;

		if (time > volume.timePerFrame)
		{
			numBlocksCopied = numBlocksExtrapolated = 0;

			if (currentTimestep < volume.timesteps - 2)
				currentTimestep++;
			else
				currentTimestep = 0;

			oldTime = currentTime;

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
				if (currentTimestep == ratioTimeSteps)
				{
					maxRatio = 0.0f;
					minRatio = 100.0f;
					meanRatio = 0.0f;
					stdDev = 0.0f;

					for (int i=2; i<ratioTimeSteps; i++)
					{
						maxRatio = glm::max(maxRatio, ratios[i]);
						minRatio = glm::min(minRatio, ratios[i]);
						meanRatio += ratios[i];

//						if (ratios[i] == 0)
//							int a =0;
					}

					meanRatio /= ratioTimeSteps;

					for (int i=2; i<ratioTimeSteps; i++)
					{
						stdDev += glm::pow((ratios[i] - meanRatio), 2.0f);
					}

					stdDev /= ratioTimeSteps;
					stdDev = glm::sqrt(stdDev);

					std::cout << "Max: " << maxRatio << std::endl;
					std::cout << "Min: " << minRatio << std::endl;
					std::cout << "Mean: " << meanRatio << std::endl;
					std::cout << "StdDev: " << stdDev << std::endl;
					getchar();
				}

				GPUPredict(volume);
				CPUPredict(volume);

				CopyChunkToGPU(volume);


//				glBindTexture(GL_TEXTURE_3D, nextTexture3D);
//				glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, volume.xRes, volume.yRes, volume.zRes, 0,  GL_RED, GL_UNSIGNED_BYTE, (volume.memblock3D + (textureSize * currentTimestep)));
//				glBindTexture(GL_TEXTURE_3D, 0);
			}
			std::cout << "Copied: " << numBlocksCopied << " - Extrapolated: " << numBlocksExtrapolated << std::endl;
			ratios[currentTimestep] = (float)numBlocksExtrapolated / (float) numBlocks;
		}
	}	
}


GLuint BlockRaycaster::GenerateTexture(VolumeDataset &volume)
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




void BlockRaycaster::Raycast(VolumeDataset &volume, TransferFunction &transferFunction, GLuint shaderProgramID, Camera &camera)
{
	int uniformLoc;

	glm::mat4 model_mat = glm::mat4(1.0f);

	uniformLoc = glGetUniformLocation (shaderProgramID, "proj");
	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &camera.projMat[0][0]);

	uniformLoc = glGetUniformLocation (shaderProgramID, "view");
	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &camera.viewMat[0][0]);

	uniformLoc = glGetUniformLocation (shaderProgramID, "model");
	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &model_mat[0][0]);

	glActiveTexture (GL_TEXTURE0);
	uniformLoc = glGetUniformLocation(shaderProgramID,"volume");
	glUniform1i(uniformLoc,0);
	glBindTexture (GL_TEXTURE_3D, currTexture3D);

	uniformLoc = glGetUniformLocation(shaderProgramID,"camPos");
	glUniform3f(uniformLoc, camera.position.x, camera.position.y, camera.position.z);

	uniformLoc = glGetUniformLocation(shaderProgramID,"maxRaySteps");
	glUniform1i(uniformLoc, maxRaySteps);

	uniformLoc = glGetUniformLocation(shaderProgramID,"rayStepSize");
	glUniform1f(uniformLoc, rayStepSize);

	uniformLoc = glGetUniformLocation(shaderProgramID,"gradientStepSize");
	glUniform1f(uniformLoc, gradientStepSize);


	uniformLoc = glGetUniformLocation(shaderProgramID,"lightPosition");
	glUniform3f(uniformLoc, lightPosition.x, lightPosition.y, lightPosition.z);

	glActiveTexture (GL_TEXTURE1);
	uniformLoc = glGetUniformLocation(shaderProgramID,"transferFunc");
	glUniform1i(uniformLoc,1);
	glBindTexture (GL_TEXTURE_1D, transferFunction.tfTexture);


	// Final render is the front faces of a cube rendered
	glBegin(GL_QUADS);

	// Front Face
	glVertex3f(-1.0f, -1.0f,  1.0f);
	glVertex3f( 1.0f, -1.0f,  1.0f);
	glVertex3f( 1.0f,  1.0f,  1.0f);
	glVertex3f(-1.0f,  1.0f,  1.0f);
 
	// Back Face
	glVertex3f(-1.0f, -1.0f, -1.0f);
	glVertex3f(-1.0f,  1.0f, -1.0f);
	glVertex3f( 1.0f,  1.0f, -1.0f);
	glVertex3f( 1.0f, -1.0f, -1.0f);
 
	// Top Face
	glVertex3f(-1.0f,  1.0f, -1.0f);
	glVertex3f(-1.0f,  1.0f,  1.0f);
	glVertex3f( 1.0f,  1.0f,  1.0f);
	glVertex3f( 1.0f,  1.0f, -1.0f);
	
	// Bottom Face
	glVertex3f(-1.0f, -1.0f, -1.0f);
	glVertex3f( 1.0f, -1.0f, -1.0f);
	glVertex3f( 1.0f, -1.0f,  1.0f);
	glVertex3f(-1.0f, -1.0f,  1.0f);
 
	// Right face
	glVertex3f( 1.0f, -1.0f, -1.0f);
	glVertex3f( 1.0f,  1.0f, -1.0f);
	glVertex3f( 1.0f,  1.0f,  1.0f);
	glVertex3f( 1.0f, -1.0f,  1.0f);
 
	// Left Face
	glVertex3f(-1.0f, -1.0f, -1.0f);
	glVertex3f(-1.0f, -1.0f,  1.0f);
	glVertex3f(-1.0f,  1.0f,  1.0f);
	glVertex3f(-1.0f,  1.0f, -1.0f);

	glEnd();

	glBindTexture(GL_TEXTURE_3D, 0);
}

// Messy but just inputs all data to shader
void BlockRaycaster::BlockRaycast(VolumeDataset &volume, TransferFunction &transferFunction, GLuint shaderProgramID, Camera &camera)
{
	int uniformLoc;

	glm::mat4 model_mat = glm::mat4(1.0f);

	uniformLoc = glGetUniformLocation (shaderProgramID, "proj");
	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &camera.projMat[0][0]);

	uniformLoc = glGetUniformLocation (shaderProgramID, "view");
	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &camera.viewMat[0][0]);

	uniformLoc = glGetUniformLocation (shaderProgramID, "model");
	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &model_mat[0][0]);

	glActiveTexture (GL_TEXTURE0);
	uniformLoc = glGetUniformLocation(shaderProgramID,"volume");
	glUniform1i(uniformLoc,0);
	glBindTexture (GL_TEXTURE_3D, currTexture3D);

	uniformLoc = glGetUniformLocation(shaderProgramID,"camPos");
	glUniform3f(uniformLoc, camera.position.x, camera.position.y, camera.position.z);

	uniformLoc = glGetUniformLocation(shaderProgramID,"maxRaySteps");
	glUniform1i(uniformLoc, maxRaySteps);

	uniformLoc = glGetUniformLocation(shaderProgramID,"rayStepSize");
	glUniform1f(uniformLoc, rayStepSize);

	uniformLoc = glGetUniformLocation(shaderProgramID,"gradientStepSize");
	glUniform1f(uniformLoc, gradientStepSize);

	uniformLoc = glGetUniformLocation(shaderProgramID,"lightPosition");
	glUniform3f(uniformLoc, lightPosition.x, lightPosition.y, lightPosition.z);

	glActiveTexture (GL_TEXTURE2);
	uniformLoc = glGetUniformLocation(shaderProgramID,"transferFunc");
	glUniform1i(uniformLoc,2);
	glBindTexture (GL_TEXTURE_1D, transferFunction.tfTexture);

//	glEnable(GL_DEPTH_TEST);
//	glBlendFunc(GL_ONE, GL_ONE);
//	glEnable(GL_BLEND);

	// Final render is the front faces of a cube rendered
	

	for (int i=0; i<numBlocks; i++)
	{
		glBegin(GL_POLYGON);
		glVertex3f(blocks[i].vertices[0].x, blocks[i].vertices[0].y, blocks[i].vertices[0].z);
		glVertex3f(blocks[i].vertices[1].x, blocks[i].vertices[1].y, blocks[i].vertices[1].z);
		glVertex3f(blocks[i].vertices[3].x, blocks[i].vertices[3].y, blocks[i].vertices[3].z);
		glVertex3f(blocks[i].vertices[2].x, blocks[i].vertices[2].y, blocks[i].vertices[2].z);
	
		glVertex3f(blocks[i].vertices[4].x, blocks[i].vertices[4].y, blocks[i].vertices[4].z);
		glVertex3f(blocks[i].vertices[5].x, blocks[i].vertices[5].y, blocks[i].vertices[5].z);
		glVertex3f(blocks[i].vertices[7].x, blocks[i].vertices[7].y, blocks[i].vertices[7].z);
		glVertex3f(blocks[i].vertices[6].x, blocks[i].vertices[6].y, blocks[i].vertices[6].z);
	
	
		glVertex3f(blocks[i].vertices[0].x, blocks[i].vertices[0].y, blocks[i].vertices[0].z);
		glVertex3f(blocks[i].vertices[1].x, blocks[i].vertices[1].y, blocks[i].vertices[1].z);
		glVertex3f(blocks[i].vertices[5].x, blocks[i].vertices[5].y, blocks[i].vertices[5].z);
		glVertex3f(blocks[i].vertices[4].x, blocks[i].vertices[4].y, blocks[i].vertices[4].z);
	
		glVertex3f(blocks[i].vertices[2].x, blocks[i].vertices[2].y, blocks[i].vertices[2].z);
		glVertex3f(blocks[i].vertices[3].x, blocks[i].vertices[3].y, blocks[i].vertices[3].z);
		glVertex3f(blocks[i].vertices[7].x, blocks[i].vertices[7].y, blocks[i].vertices[7].z);
		glVertex3f(blocks[i].vertices[6].x, blocks[i].vertices[6].y, blocks[i].vertices[6].z);
	
	
		glVertex3f(blocks[i].vertices[0].x, blocks[i].vertices[0].y, blocks[i].vertices[0].z);
		glVertex3f(blocks[i].vertices[2].x, blocks[i].vertices[2].y, blocks[i].vertices[2].z);
		glVertex3f(blocks[i].vertices[6].x, blocks[i].vertices[6].y, blocks[i].vertices[6].z);
		glVertex3f(blocks[i].vertices[4].x, blocks[i].vertices[4].y, blocks[i].vertices[4].z);

		glVertex3f(blocks[i].vertices[1].x, blocks[i].vertices[1].y, blocks[i].vertices[1].z);
		glVertex3f(blocks[i].vertices[3].x, blocks[i].vertices[3].y, blocks[i].vertices[3].z);
		glVertex3f(blocks[i].vertices[7].x, blocks[i].vertices[7].y, blocks[i].vertices[7].z);
		glVertex3f(blocks[i].vertices[5].x, blocks[i].vertices[5].y, blocks[i].vertices[5].z);
		glEnd();
	}

	

	glBindTexture(GL_TEXTURE_3D, 0);
}


/*
bool BlockRaycaster::BlockCompare(VolumeDataset &volume, int x, int y, int z)
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
*/