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
	cudaLMSE.resize(numPixels);
	HANDLE_ERROR( cudaMalloc((void**)&cudaNumZeroLaplacians, sizeof(int)) );
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
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, xPixels, yPixels, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);		// CHECK THIS
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
texture <uchar1, cudaTextureType2D, cudaReadModeElementType> extrapTexRef;


__global__ void CudaCompare(int numPixels, int xPixels, int yPixels, thrust::device_ptr<float> cudaMSE, thrust::device_ptr<float> cudaMAE, thrust::device_ptr<float> cudaLMSE, int *numZeroLaplacians)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < numPixels)
	{
		int v = (int) tid / yPixels;
		int u = tid % yPixels;

		uchar1 bruteColor = tex2D(bruteTexRef, u, v);
		uchar1 extrapColor = tex2D(extrapTexRef, u, v);

		float diff = bruteColor.x - extrapColor.x;

		cudaMSE[tid] = diff * diff;
		cudaMAE[tid] = glm::abs(diff);

		if (u == 0 || u == xPixels-1 || v == 0 || v == yPixels-1)
		{
			cudaLMSE[tid] = 0.0f;
			return;
		}

		float bruteLaplace = tex2D(bruteTexRef, u+1, v).x + tex2D(bruteTexRef, u-1, v).x + tex2D(bruteTexRef, u, v+1).x + tex2D(bruteTexRef, u, v-1).x - (4 * bruteColor.x);
		float extrapLaplace = tex2D(extrapTexRef, u+1, v).x + tex2D(extrapTexRef, u-1, v).x + tex2D(extrapTexRef, u, v+1).x + tex2D(extrapTexRef, u, v-1).x - (4 * extrapColor.x);

		float result = glm::pow(bruteLaplace - extrapLaplace, 2.0f) / glm::pow(bruteLaplace, 2.0f);

		if (bruteLaplace != 0.0f)
			cudaLMSE[tid] = result;
		else
		{
			cudaLMSE[tid] = 0.0f;
			atomicAdd(numZeroLaplacians, (int)1);
		}

//		if (bruteLaplace != 0.0f && result != 0.0f)
//			printf("%f, %f, %f\n", bruteLaplace, extrapLaplace, result);
	}
}


void ErrorMetrics::CompareImages()
{
	HANDLE_ERROR( cudaMemset(cudaNumZeroLaplacians, 0, sizeof(int)) );

	HANDLE_ERROR( cudaGraphicsGLRegisterImage(&cudaResources[0], bruteImage, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone) );
	HANDLE_ERROR( cudaGraphicsGLRegisterImage(&cudaResources[1], interpImage, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone) );
	HANDLE_ERROR( cudaGraphicsMapResources(2, &cudaResources[0]) );

	cudaArray *bruteArray = 0;	
	HANDLE_ERROR( cudaGraphicsSubResourceGetMappedArray(&bruteArray, cudaResources[0], 0, 0) ); 
	HANDLE_ERROR( cudaBindTextureToArray(bruteTexRef, bruteArray) );

	cudaArray *extrapArray = 0;	
	HANDLE_ERROR( cudaGraphicsSubResourceGetMappedArray(&extrapArray, cudaResources[1], 0, 0) ); 
	HANDLE_ERROR( cudaBindTextureToArray(extrapTexRef, extrapArray) );



	CudaCompare<<<(numPixels + 255) / 256, 256>>>(numPixels, xPixels, yPixels, &cudaMSE[0], &cudaMAE[0], &cudaLMSE[0], cudaNumZeroLaplacians);


	HANDLE_ERROR( cudaUnbindTexture(bruteTexRef) );
	HANDLE_ERROR( cudaUnbindTexture(extrapTexRef) );

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

struct Is_Non_Negative 
{ 
	__host__ __device__ bool operator()(const float& x) const 
	{ 
		return x >= 0.0f; 
	} 
};

void ErrorMetrics::GetErrorMetrics()
{
	meanSqrError = thrust::reduce(cudaMSE.begin(), cudaMSE.end(), 0.0f, thrust::plus<float>());
	meanSqrError /= (float)numPixels;

	meanAvgErr = thrust::reduce(cudaMAE.begin(), cudaMAE.end(), 0.0f, thrust::plus<float>());
	meanAvgErr /= (float)numPixels;

	int numZeroLaplacians;
	HANDLE_ERROR( cudaMemcpy(&numZeroLaplacians, cudaNumZeroLaplacians, sizeof(int), cudaMemcpyDeviceToHost) );
	int numValid = ((xPixels - 2) * (yPixels - 2)) - numZeroLaplacians;
	laplaceMSE = thrust::reduce(cudaLMSE.begin(), cudaLMSE.end(), 0.0f, thrust::plus<float>());
	laplaceMSE /= (float)numValid;

	peakSigToNoise = 10.0f * log10f((255.0f * 255.0f) / meanSqrError);

	maxDifference = thrust::reduce(cudaMAE.begin(), cudaMAE.end(), 0.0f, Max_Functor());

//	std::cout << "Mean Square Error: " << meanSqrError << std::endl;
//	std::cout << "Mean Average Error: " << meanAvgErr << std::endl;
//	std::cout << "Peak Signal To Noise: " << peakSigToNoise << std::endl;
//	std::cout << "Max Difference: " << maxDifference << std::endl;
}


ErrorMetrics::~ErrorMetrics()
{
	cudaMSE.clear();
	cudaMAE.clear();
	cudaLMSE.clear();
}