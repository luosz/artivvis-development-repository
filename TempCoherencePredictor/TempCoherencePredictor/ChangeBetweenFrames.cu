#include "ChangeBetweenFrames.h"

void ChangeBetweenFrames::Init(VolumeDataset &volume)
{
	cudaResources.push_back(cudaGraphicsResource_t());
	cudaResources.push_back(cudaGraphicsResource_t());

	prevTexture3D = Generate3DTexture(volume);

	HANDLE_ERROR( cudaMalloc((void**)&l1, sizeof(float)) );
	HANDLE_ERROR( cudaMalloc((void**)&l2, sizeof(float)) );
	HANDLE_ERROR( cudaMalloc((void**)&l3, sizeof(float)) );
	HANDLE_ERROR( cudaMalloc((void**)&l4, sizeof(float)) );
	HANDLE_ERROR( cudaMalloc((void**)&l5, sizeof(float)) );
}


GLuint ChangeBetweenFrames::Generate3DTexture(VolumeDataset &volume)
{
	GLuint tex;

	texture3DSize = volume.xRes * volume.yRes * volume.zRes * volume.bytesPerElement;

	glEnable(GL_TEXTURE_3D);
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_3D, tex);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, volume.xRes, volume.yRes, volume.zRes, 0,  GL_RED, GL_UNSIGNED_BYTE, volume.memblock3D);

	glBindTexture(GL_TEXTURE_3D, 0);

	return tex;
}

texture <unsigned char, cudaTextureType3D, cudaReadModeElementType> prevTexRef;
texture <unsigned char, cudaTextureType3D, cudaReadModeElementType> currTexRef;

__global__ void CudaFindDiffBetweenFrames(int numVoxels, int xRes, int yRes, int zRes, float *l1, float *l2, float *l3, float *l4, float *l5)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < numVoxels)
	{
		int z = tid / (xRes * yRes);
		int remainder = tid % (xRes * yRes);

		int y = remainder / xRes;

		int x = remainder % xRes;

		unsigned char prevVal, currVal;

		prevVal = tex3D(prevTexRef, x, y, z);
		currVal = tex3D(currTexRef, x, y, z);

		float percent = (float)glm::abs(prevVal - currVal);

		if (prevVal != (unsigned char)0)
			percent *= (100.0f / (float)prevVal);
		else
			percent *= 100.0f;

	//	printf ("%u - %u - %f\n", prevVal, currVal, percent);

		if (percent < 0.5f)
			atomicAdd(l1, (float)1.0f);
		else if (percent < 1.0f)
			atomicAdd(l2, (float)1.0f);
		else if (percent < 3.0f)
			atomicAdd(l3, (float)1.0f);
		else if (percent < 10.0f)
			atomicAdd(l4, (float)1.0f);
		else
			atomicAdd(l5, (float)1.0f);
	}
}

void ChangeBetweenFrames::Find(VolumeDataset &volume, int currentTimestep, GLuint bruteTex3D)
{
	HANDLE_ERROR( cudaMemset(l1, 0, sizeof(float)) );
	HANDLE_ERROR( cudaMemset(l2, 0, sizeof(float)) );
	HANDLE_ERROR( cudaMemset(l3, 0, sizeof(float)) );
	HANDLE_ERROR( cudaMemset(l4, 0, sizeof(float)) );
	HANDLE_ERROR( cudaMemset(l5, 0, sizeof(float)) );


	glBindTexture(GL_TEXTURE_3D, prevTexture3D);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, volume.xRes, volume.yRes, volume.zRes, 0,  GL_RED, GL_UNSIGNED_BYTE, (volume.memblock3D + (texture3DSize * (currentTimestep-1))));
	glBindTexture(GL_TEXTURE_3D, 0);

	HANDLE_ERROR( cudaGraphicsGLRegisterImage(&cudaResources[0], prevTexture3D, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsNone) );
	HANDLE_ERROR( cudaGraphicsGLRegisterImage(&cudaResources[1], bruteTex3D, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsNone) );
	HANDLE_ERROR( cudaGraphicsMapResources(2, &cudaResources[0]) );

	cudaArray *prevArry = 0;	
	HANDLE_ERROR( cudaGraphicsSubResourceGetMappedArray(&prevArry, cudaResources[0], 0, 0) ); 
	HANDLE_ERROR( cudaBindTextureToArray(prevTexRef, prevArry) );

	cudaArray *currArry = 0;	
	HANDLE_ERROR( cudaGraphicsSubResourceGetMappedArray(&currArry, cudaResources[1], 0, 0) ); 
	HANDLE_ERROR( cudaBindTextureToArray(currTexRef, currArry) );

	CudaFindDiffBetweenFrames <<<(volume.numVoxels + 255) / 256, 256>>> (volume.numVoxels, volume.xRes, volume.yRes, volume.zRes, l1, l2, l3, l4, l5);

	HANDLE_ERROR( cudaUnbindTexture(prevTexRef) );
	HANDLE_ERROR( cudaUnbindTexture(currTexRef) );

	HANDLE_ERROR( cudaGraphicsUnmapResources(2, &cudaResources[0]) );
	HANDLE_ERROR( cudaGraphicsUnregisterResource(cudaResources[0]) );
	HANDLE_ERROR( cudaGraphicsUnregisterResource(cudaResources[1]) );

	HANDLE_ERROR( cudaMemcpy(&la1, l1, sizeof(float), cudaMemcpyDeviceToHost) );
	HANDLE_ERROR( cudaMemcpy(&la2, l2, sizeof(float), cudaMemcpyDeviceToHost) );
	HANDLE_ERROR( cudaMemcpy(&la3, l3, sizeof(float), cudaMemcpyDeviceToHost) );
	HANDLE_ERROR( cudaMemcpy(&la4, l4, sizeof(float), cudaMemcpyDeviceToHost) );
	HANDLE_ERROR( cudaMemcpy(&la5, l5, sizeof(float), cudaMemcpyDeviceToHost) );

	la1 /= volume.numVoxels;
	la2 /= volume.numVoxels;
	la3 /= volume.numVoxels;
	la4 /= volume.numVoxels;
	la5 /= volume.numVoxels;

//	std::cout << la1 << " - " << la2 << " - " << la3 << " - " << la4 << " - " << la5 << std::endl;
}
