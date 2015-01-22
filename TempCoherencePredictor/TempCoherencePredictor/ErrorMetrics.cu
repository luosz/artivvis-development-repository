#include "ErrorMetrics.h"

void ErrorMetrics::Init(int screenWidth, int screenHeight)
{
	xPixels = screenWidth;
	yPixels = screenHeight;
	numPixels = xPixels * yPixels;

	// Generate two textures for alternating read and write on framebuffer
	bruteImage = Generate2DTexture();
	interpImage = Generate2DTexture();

	framebuffer.Generate(screenWidth, screenHeight, bruteImage);

	cudaResources.push_back(cudaGraphicsResource_t());
	cudaResources.push_back(cudaGraphicsResource_t());

	cudaMSE.resize(numPixels);
	cudaMAE.resize(numPixels);
}

GLuint ErrorMetrics::Generate2DTexture()
{
	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, xPixels, yPixels, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL); 
	glBindTexture(GL_TEXTURE_2D, 0);

	return tex;
}


void ErrorMetrics::FindError(TransferFunction &transferFunction, ShaderManager &shaderManager, Camera &camera, Raycaster &raycaster, GLuint bruteTex3D, GLuint interpTex3D)
{
	RenderImages(transferFunction, shaderManager, camera, raycaster, bruteTex3D, interpTex3D);
	CompareImages();
	GetErrorMetrics();
}


texture <uchar1, cudaTextureType2D, cudaReadModeElementType> bruteTexRef;
texture <uchar1, cudaTextureType2D, cudaReadModeElementType> interpTexRef;


__global__ void CudaCompare(int numPixels, int xPixels, int yPixels, thrust::device_ptr<float> cudaMSE, thrust::device_ptr<float> cudaMAE)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < numPixels)
	{
		int v = (int) tid / yPixels;
		int u = tid % yPixels;

		uchar1 bruteColor = tex2D(bruteTexRef, u, v);
		uchar1 interpColor = tex2D(interpTexRef, u, v);

		float diff = bruteColor.x - interpColor.x;

		cudaMSE[tid] = diff * diff;
		cudaMAE[tid] = glm::abs(diff);

//		if (diff > 0.0f)
//			printf("%d\n", diff);
	}
}


void ErrorMetrics::CompareImages()
{
	HANDLE_ERROR( cudaGraphicsGLRegisterImage(&cudaResources[0], bruteImage, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone) );
	HANDLE_ERROR( cudaGraphicsGLRegisterImage(&cudaResources[1], interpImage, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone) );
	HANDLE_ERROR( cudaGraphicsMapResources(2, &cudaResources[0]) );

	cudaArray *bruteArray = 0;	
	HANDLE_ERROR( cudaGraphicsSubResourceGetMappedArray(&bruteArray, cudaResources[0], 0, 0) ); 
	HANDLE_ERROR( cudaBindTextureToArray(bruteTexRef, bruteArray) );

	cudaArray *interpArray = 0;	
	HANDLE_ERROR( cudaGraphicsSubResourceGetMappedArray(&interpArray, cudaResources[1], 0, 0) ); 
	HANDLE_ERROR( cudaBindTextureToArray(interpTexRef, interpArray) );



	CudaCompare<<<(numPixels + 255) / 256, 256>>>(numPixels, xPixels, yPixels, &cudaMSE[0], &cudaMAE[0]);


	HANDLE_ERROR( cudaUnbindTexture(bruteTexRef) );
	HANDLE_ERROR( cudaUnbindTexture(interpTexRef) );

	HANDLE_ERROR( cudaGraphicsUnmapResources(2, &cudaResources[0]) );
	HANDLE_ERROR( cudaGraphicsUnregisterResource(cudaResources[0]) );
	HANDLE_ERROR( cudaGraphicsUnregisterResource(cudaResources[1]) );
}


void ErrorMetrics::RenderImages(TransferFunction &transferFunction, ShaderManager &shaderManager, Camera &camera, Raycaster &raycaster, GLuint bruteTex3D, GLuint interpTex3D)
{
	GLuint shaderProgramID = shaderManager.UseShader(TFShader);

	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer.ID);

	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, bruteImage, 0);
	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	raycaster.Raycast(transferFunction, shaderProgramID, camera, bruteTex3D);

	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, interpImage, 0);
	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	raycaster.Raycast(transferFunction, shaderProgramID, camera, interpTex3D);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


struct Max_Functor 
{ 
	__host__ __device__ float operator()(const float& x, const float& y) const 
	{ 
		return glm::max(glm::abs(x), glm::abs(y)); 
	} 
};

void ErrorMetrics::GetErrorMetrics()
{
	meanSqrError = thrust::reduce(cudaMSE.begin(), cudaMSE.end(), 0.0f, thrust::plus<float>());
	meanSqrError /= numPixels;

	meanAvgErr = thrust::reduce(cudaMAE.begin(), cudaMAE.end(), 0.0f, thrust::plus<float>());
	meanAvgErr /= numPixels;

	peakSigToNoise = 10.0f * log10f((255.0f * 255.0f) / meanSqrError);

	maxDifference = thrust::reduce(cudaMAE.begin(), cudaMAE.end(), 0.0f, Max_Functor());

//	std::cout << "Mean Square Error: " << meanSqrError << std::endl;
//	std::cout << "Mean Average Error: " << meanAvgErr << std::endl;
//	std::cout << "Peak Signal To Noise: " << peakSigToNoise << std::endl;
//	std::cout << "Max Difference: " << maxDifference << std::endl;
}