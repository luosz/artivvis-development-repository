#include "BlockRaycaster.h"

texture <unsigned char, cudaTextureType3D, cudaReadModeElementType> texRef;

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


}

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
					scalar = tex3D(texRef, xMin + i, yMin + j, zMin + k);
				}

		
//		scalar = tex3D(texRef, 50, 50, 50);
//		if (scalar < 256 && scalar > 0)
//			printf("%d: %u\n", tid, scalar);
	}
}

void BlockRaycaster::GPUPredict(VolumeDataset &volume)
{
	HANDLE_ERROR( cudaGraphicsGLRegisterImage(&resource, volume.prevTexture3D, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsNone) );
	HANDLE_ERROR( cudaGraphicsMapResources(1, &resource) );
	cudaArray *arry = 0;	
	HANDLE_ERROR( cudaGraphicsSubResourceGetMappedArray(&arry, resource, 0, 0) ); 
	HANDLE_ERROR( cudaBindTextureToArray(texRef, arry) );

	CudaPredict <<<(numBlocks + 255) / 256, 256>>>(numBlocks, numXBlocks, numYBlocks, numZBlocks, blockRes);

	// Unbind and unmap, must be done before OpenGL uses texture memory again
	HANDLE_ERROR( cudaUnbindTexture(texRef) );
	HANDLE_ERROR( cudaGraphicsUnmapResources(1, &resource) );
	HANDLE_ERROR( cudaGraphicsUnregisterResource(resource) );
}


void BlockRaycaster::CPUPredict(VolumeDataset &volume)
{
	int prevTimestep;

	if (volume.currentTimestep == 0)
		prevTimestep = 0;
	else
		prevTimestep = volume.currentTimestep - 1;

	GLubyte *volumeStart = volume.memblock3D + (prevTimestep * volume.xRes *volume.yRes * volume.zRes * volume.bytesPerElement);

	for (int z=0; z<numZBlocks; z++)
		for (int y =0; y<numYBlocks; y++)
			for (int x=0; x<numXBlocks; x++)
			{
				int xMin = x * blockRes;
				int yMin = y * blockRes;
				int zMin = z * blockRes;
				unsigned char scalar;

				for (int k=0; k<blockRes; k++)
					for (int j=0; j<blockRes; j++)
						for (int i=0; i<blockRes; i++)
						{
							scalar = volumeStart[(xMin + i) + ((yMin + j) * volume.xRes) + ((zMin + k) * volume.xRes * volume.yRes)];
						}
			}
			 

//	scalar = volumeStart[50 + (50 * volume.xRes) + (50 * volume.xRes * volume.yRes)];
//	std::cout << "CPU: " << (int)scalar << std::endl;
}


void BlockRaycaster::TemporalCoherence(VolumeDataset &volume)
{
	GPUPredict(volume);
	CPUPredict(volume);
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
	glBindTexture (GL_TEXTURE_3D, volume.currTexture3D);

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
	glBindTexture (GL_TEXTURE_3D, volume.currTexture3D);

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
