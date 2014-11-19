#include "RegionVisibilityOptimizer.h"

void RegionVisibilityOptimizer::Init(TransferFunction &transferFunction)
{	
	numRegions = 4;
	xPixels = 800;
	yPixels = 800;

	glGenTextures(1, &bufferTex);
    glBindTexture(GL_TEXTURE_2D, bufferTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, xPixels, yPixels, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL); 

	glBindTexture(GL_TEXTURE_2D, 0);


	glGenFramebuffers (1, &frameBuffer);
	glBindFramebuffer (GL_FRAMEBUFFER, frameBuffer);

	unsigned int rb = 0;
	glGenRenderbuffers (1, &rb);
	glBindRenderbuffer (GL_RENDERBUFFER, rb);
	glRenderbufferStorage (GL_RENDERBUFFER, GL_DEPTH_COMPONENT, xPixels, yPixels);
	glFramebufferRenderbuffer (GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb);

	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, bufferTex, 0);

	glBindFramebuffer (GL_FRAMEBUFFER, 0);

	HANDLE_ERROR( cudaMalloc((void**)&cudaRegionVisibilities, numRegions * sizeof(float)) );
	HANDLE_ERROR( cudaMalloc((void**)&cudaNumInRegion, numRegions * sizeof(float)) );
	regionVisibilities.resize(numRegions);
}


texture <float4, cudaTextureType2D, cudaReadModeElementType> texRef2;

__global__ void CudaRegionEvaluate(int xPixels, int yPixels, float *cudaRegionVisibilities, int *cudaNumInRegion)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < xPixels * yPixels)
	{
		int v = (int) tid / yPixels;
		int u = tid % yPixels;

		float4 color = tex2D(texRef2, u, v);

//		if (u == 400 && v == 400)
//				printf("%f, %f, %f, %f\n", color.x, color.y, color.z, color.w);

		if (color.x > 0.01f)
		{
			atomicAdd(&(cudaRegionVisibilities[0]), (float)color.x);
			atomicAdd(&(cudaNumInRegion[0]), (int)1);
		}

		if (color.y > 0.01f)
		{
			atomicAdd(&(cudaRegionVisibilities[1]), (float)color.x);
			atomicAdd(&(cudaNumInRegion[1]), (int)1);
		}

		if (color.z > 0.01f)
		{
			atomicAdd(&(cudaRegionVisibilities[2]), (float)color.x);
			atomicAdd(&(cudaNumInRegion[2]), (int)1);

//			if (color.z > 0.4f)
//				printf("%d, %d, %f\n", u, v, color.z);
		}

		if (color.w > 0.01f)
		{
			atomicAdd(&(cudaRegionVisibilities[3]), (float)color.x);
			atomicAdd(&(cudaNumInRegion[3]), (int)1);
		}
	}
}


__global__ void CudaRegionNormalize(int numBins, float *cudaRegionVisibilities, int *cudaNumInRegion)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < numBins)
	{
		if (cudaNumInRegion[tid] > 0)
		{
		//	if (tid == 44)
		//		printf("%d, %f\n", numInBin[tid], histBins[tid]);
			
	
			cudaRegionVisibilities[tid] = cudaRegionVisibilities[tid] / (cudaNumInRegion[tid]);
	
			printf("%d: %f, %d\n", tid, cudaRegionVisibilities[tid], cudaNumInRegion[tid]);
		
		}
	}
}


void RegionVisibilityOptimizer::CalculateVisibility(ShaderManager &shaderManager, Camera &camera, VolumeDataset &volume, TransferFunction &transferFunction,  Raycaster *raycaster)
{
	HANDLE_ERROR( cudaMemset(cudaRegionVisibilities, 0, numRegions * sizeof(float)) );
	HANDLE_ERROR( cudaMemset(cudaNumInRegion, 0, numRegions * sizeof(int)) );

	GLuint shaderProgramID = shaderManager.UseShader(RegionVisibilityShader);

	glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, bufferTex, 0);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	raycaster->Raycast(volume, transferFunction, shaderProgramID, camera);


	HANDLE_ERROR( cudaGraphicsGLRegisterImage(&resource, bufferTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone) );
	HANDLE_ERROR( cudaGraphicsMapResources(1, &resource) );
	cudaArray *arry = 0;	
	HANDLE_ERROR( cudaGraphicsSubResourceGetMappedArray(&arry, resource, 0, 0) ); 
	HANDLE_ERROR( cudaBindTextureToArray(texRef2, arry) );

	int numPixels = xPixels * yPixels;
	CudaRegionEvaluate<<<(numPixels + 255) / 256, 256>>>(xPixels, yPixels, cudaRegionVisibilities, cudaNumInRegion);

	HANDLE_ERROR( cudaUnbindTexture(texRef2) );
	HANDLE_ERROR( cudaGraphicsUnmapResources(1, &resource) );
	HANDLE_ERROR( cudaGraphicsUnregisterResource(resource) );

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	CudaRegionNormalize<<< (numRegions + 255) / 256, 256>>>(numRegions, cudaRegionVisibilities, cudaNumInRegion);

	HANDLE_ERROR( cudaMemcpy(&regionVisibilities[0], cudaRegionVisibilities, numRegions * sizeof(float), cudaMemcpyDeviceToHost) );
	cudaDeviceSynchronize();
}