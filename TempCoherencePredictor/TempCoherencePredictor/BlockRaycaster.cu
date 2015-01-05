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

	blockRes = 8;
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

	prevTempVolume = new GLubyte[volume.numVoxels * volume.bytesPerElement];
	currTempVolume = new GLubyte[volume.numVoxels * volume.bytesPerElement];
	nextTempVolume = new GLubyte[volume.numVoxels * volume.bytesPerElement];

	epsilon = 3;
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

//		surf3Dread(&temp1, surface, x, y, z);

		nextVal = (2 * currVal) - prevVal;

		surf3Dwrite(nextVal, surface, x, y, z);

//		if (x == 60 && y == 5 && z == 60)
//			printf("%u, %u, %u\n", temp1, nextVal, temp2);
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
//	HANDLE_ERROR( cudaBindTextureToArray(nextTexRef, nextArry) );

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
	bool similar = true;

	GLubyte *nextVolume = volume.memblock3D + (currentTimestep * volume.numVoxels);

	int xMin = x * blockRes;
	int yMin = y * blockRes;
	int zMin = z * blockRes;

	unsigned char oldVal, currVal;
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
				{
					similar = false;
					currTempVolume[ID] = n;
//					return false;
				}
//				else if ((absDiff > 0) && (n <= 1))
//				{
//					similar = false;
//					currTempVolume[ID] = n;
////					return false;
//				}

			}

	return similar;
}


void BlockRaycaster::CPUPredict(VolumeDataset &volume)
{
	int prevTimestep = currentTimestep - 2;
	int currTimestep = currentTimestep - 1;

	GLubyte *prevVolume = volume.memblock3D + (prevTimestep * volume.numVoxels);
	GLubyte *currVolume = volume.memblock3D + (currTimestep * volume.numVoxels);

	for (int i=0; i<volume.numVoxels; i++)
	{
		
//		tempVolume[i] = (2 * prevVolume[i]) - tempVolume[i];

		nextTempVolume[i] = (2 * currTempVolume[i]) - prevTempVolume[i];

		prevTempVolume[i] = currTempVolume[i];
		currTempVolume[i] = nextTempVolume[i];
	}

	glBindTexture(GL_TEXTURE_3D, nextTexture3D);

	for (int z=0; z<numZBlocks; z++)
		for (int y =0; y<numYBlocks; y++)
			for (int x=0; x<numXBlocks; x++)
			{
				if (BlockCompare(volume, x, y, z) == false)
				{
					for (int k=0; k<blockRes; k++)
						for (int j=0; j<blockRes; j++)
						{
							GLubyte *mainBlockAddress = volume.memblock3D + (currentTimestep * volume.numVoxels) + (x * blockRes) + (((y * blockRes) + j) * volume.xRes) + (((z * blockRes) + k) * volume.xRes * volume.yRes);

							glTexSubImage3D(GL_TEXTURE_3D, 0, (x * blockRes), (y * blockRes) + j, (z * blockRes) + k, blockRes, 1, 1, GL_RED, GL_UNSIGNED_BYTE, mainBlockAddress);

//							GLubyte *tempBlockAddress = currTempVolume + (x * blockRes) + (((y * blockRes) + j) * volume.xRes) + (((z * blockRes) + k) * volume.xRes * volume.yRes);
//							memcpy(tempBlockAddress, mainBlockAddress, blockRes * sizeof(GLubyte));
						}
				}

			}
	
	glBindTexture(GL_TEXTURE_3D, 0);	 

//	scalar = volumeStart[50 + (50 * volume.xRes) + (50 * volume.xRes * volume.yRes)];
//	std::cout << "CPU: " << (int)scalar << std::endl;
}


void BlockRaycaster::TemporalCoherence(VolumeDataset &volume)
{
	if (volume.timesteps > 1)
	{
		clock_t currentTime = clock();
		float time = (currentTime - oldTime) / (float) CLOCKS_PER_SEC;

		if (time > volume.timePerFrame)
		{
			if (currentTimestep < volume.timesteps - 1)
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

//				tempVolume = volume.memblock3D + (textureSize * currentTimestep);

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
				GPUPredict(volume);
				CPUPredict(volume);
			}
			
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


void BlockRaycaster::UpdateTexture(VolumeDataset &volume)
{
	glDeleteTextures(1, &prevTexture3D);
	prevTexture3D = currTexture3D;

	currTexture3D = GenerateTexture(volume);
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
__global__ void CudaPredict(int numBlocks, int xBlocks, int yBlocks, int zBlocks, int blockRes)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < numBlocks)
	{
		int z = tid / (xBlocks * yBlocks);
		int remainder = tid % (xBlocks * yBlocks);

		int y = remainder / xBlocks;

		int x = remainder % xBlocks;

		int xMin = x * blockRes;
		int yMin = y * blockRes;
		int zMin = z * blockRes;

		unsigned char scalar;

		for (int k=0; k<blockRes; k++)
			for (int j=0; j<blockRes; j++)
				for (int i=0; i<blockRes; i++)
				{
					scalar = tex3D(prevTexRef, xMin + i, yMin + j, zMin + k);
				}

		
//		scalar = tex3D(texRef, 50, 50, 50);
//		if (scalar < 256 && scalar > 0)
//			printf("%d: %u\n", tid, scalar);
	}
}
*/