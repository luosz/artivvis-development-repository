#include "RegionVisibilityOptimizer.h"

RegionVisibilityOptimizer::RegionVisibilityOptimizer(VolumeDataset *volume_, TransferFunction *transferFunction_, Raycaster *raycaster_, ShaderManager *shaderManager_, Camera *camera_)
{	
	volume = volume_;
	transferFunction = transferFunction_;
	raycaster = raycaster_;
	shaderManager = shaderManager_;
	camera = camera_;

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

	glGenRenderbuffers (1, &renderBuffer);
	glBindRenderbuffer (GL_RENDERBUFFER, renderBuffer);
	glRenderbufferStorage (GL_RENDERBUFFER, GL_DEPTH_COMPONENT, xPixels, yPixels);
	glFramebufferRenderbuffer (GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, renderBuffer);

	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, bufferTex, 0);

	glBindFramebuffer (GL_FRAMEBUFFER, 0);

	HANDLE_ERROR( cudaMalloc((void**)&cudaRegionVisibilities, numRegions * sizeof(float)) );
	HANDLE_ERROR( cudaMalloc((void**)&cudaNumInRegion, numRegions * sizeof(float)) );
	regionVisibilities.resize(numRegions);
}

RegionVisibilityOptimizer::~RegionVisibilityOptimizer()
{
	glDeleteTextures(1, &bufferTex);
	glDeleteFramebuffers(1, &frameBuffer);
	glDeleteRenderbuffers(1, &renderBuffer);
	HANDLE_ERROR( cudaFree(&cudaRegionVisibilities));
	HANDLE_ERROR( cudaFree(&cudaNumInRegion));
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

		if (color.x > 0.0f)
		{
			atomicAdd(&(cudaRegionVisibilities[0]), (float)color.x);
			atomicAdd(&(cudaNumInRegion[0]), (int)1);
		}

		if (color.y > 0.0f)
		{
			atomicAdd(&(cudaRegionVisibilities[1]), (float)color.y);
			atomicAdd(&(cudaNumInRegion[1]), (int)1);
		}

		if (color.z > 0.0f)
		{
			atomicAdd(&(cudaRegionVisibilities[2]), (float)color.z);
			atomicAdd(&(cudaNumInRegion[2]), (int)1);
		}

		if (color.w > 0.0f)
		{
			atomicAdd(&(cudaRegionVisibilities[3]), (float)color.w);
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
			cudaRegionVisibilities[tid] = cudaRegionVisibilities[tid] / (cudaNumInRegion[tid]);	
		}
	}
}

void RegionVisibilityOptimizer::Optimize()
{
	CalculateVisibility();
}

void RegionVisibilityOptimizer::CalculateVisibility()
{
	HANDLE_ERROR( cudaMemset(cudaRegionVisibilities, 0, numRegions * sizeof(float)) );
	HANDLE_ERROR( cudaMemset(cudaNumInRegion, 0, numRegions * sizeof(int)) );

	GLuint shaderProgramID = shaderManager->UseShader(RegionVisibilityShader);

	glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, bufferTex, 0);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	raycaster->Raycast(*volume, *transferFunction, shaderProgramID, *camera);


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



void RegionVisibilityOptimizer::Draw(ShaderManager &shaderManager, Camera &camera)
{
	GLuint shaderProgramID = shaderManager.UseShader(SimpleShader);

	int uniformLoc;

	glm::mat4 model_mat = glm::mat4(1.0f);

	uniformLoc = glGetUniformLocation (shaderProgramID, "proj");
	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &camera.projMat[0][0]);

	uniformLoc = glGetUniformLocation (shaderProgramID, "view");
	glm::mat4 tempView = glm::lookAt(glm::vec3(0.5f, 0.5f, 2.5f), glm::vec3(0.5f, 0.5f, 0.5f), glm::vec3(0.0f,1.0f,0.0f));
	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &tempView[0][0]);

	uniformLoc = glGetUniformLocation (shaderProgramID, "model");
	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &model_mat[0][0]);


	glBegin(GL_LINES);

	glColor3f(1.0f, 1.0f, 1.0f);
	glVertex3f( -0.5f, -0.5f, 0.5f);
	glVertex3f( -0.5f, 0.5f, 0.5f);

	glVertex3f(-0.5f, -0.5f, 0.5f);
	glVertex3f(0.5f, -0.5f, 0.5f);

	glEnd();


	glBegin(GL_QUADS);

	
	for (int i=0; i<numRegions; i++)
	{
		if (i == 0)
			glColor3f(0.75f, 0.0f, 1.0f);
		else if (i == 1)
			glColor3f(1.0f, 0.0f, 0.0f);
		else if (i == 2)
			glColor3f(0.0f, 1.0f, 0.0f);

		glVertex3f((i / 4.0f) - 0.5f, -0.5f, 0.5f);
		glVertex3f((i / 4.0f) - 0.5f, regionVisibilities[i] - 0.5f, 0.5f);
		glVertex3f(((i+1) / 4.0f) - 0.5f, regionVisibilities[i] - 0.5f, 0.5f);
		glVertex3f(((i+1) / 4.0f) - 0.5f, -0.5f, 0.5f);
	}

	glEnd();
}


