#include "ImageProcessor.h"

texture <float4, cudaTextureType2D, cudaReadModeElementType> texRef;

void ImageProcessor::Init(int screenWidth, int screenHeight)
{
	xPixels = screenWidth;
	yPixels = screenHeight;
	numPixels = xPixels * yPixels;

	fbTex = new Texture2D(GL_RGBA32F, xPixels, yPixels, GL_RGBA, GL_FLOAT, NULL);
	framebuffer = new Framebuffer(xPixels, yPixels, fbTex->ID);

//	HANDLE_ERROR (cudaMalloc((void**)&cudaSquareDiffs, xPixels * yPixels * sizeof(float)) );

	aSquareDiffs.resize(numPixels);
	bSquareDiffs.resize(numPixels);
	pixelVals.resize(numPixels);
}


void ImageProcessor::Begin()
{
	framebuffer->Bind();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

/*
__global__ void CudaGetSquareDiffs(int numPixels, int xPixels, int yPixels, int aKernelRadius, thrust::device_ptr<float> aSquareDiffs, int bKernelRadius, thrust::device_ptr<float> bSquareDiffs)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < numPixels)
	{
		int v = (int) tid / xPixels;
		int u = tid % xPixels;

		float4 centre = tex2D(texRef, u, v);
		float centreVal = (centre.x + centre.y + centre.z) / 3.0f;

		float aSqDiff = 0.0f;
		float bSqDiff = 0.0f;

		for (int i = u - bKernelRadius; i <= u + bKernelRadius; i++)
		{
			for (int j = v - bKernelRadius; j <= v + bKernelRadius; j++)
			{
				float dist2 = glm::distance2(glm::vec2(i, j), glm::vec2(u, v));

				if (dist2 > bKernelRadius * bKernelRadius)
					continue;

				float4 sample = tex2D(texRef, i, j);
				float sampleVal = (sample.x + sample.y + sample.z) / 3.0f;

				float sqDiff = (centreVal - sampleVal) * (centreVal - sampleVal);

				bSqDiff += sqDiff;

				if (dist2 < aKernelRadius * aKernelRadius)
					aSqDiff += sqDiff;
			}
		}

		aSqDiff = glm::sqrt(aSqDiff);
		bSqDiff = glm::sqrt(bSqDiff);

		aSquareDiffs[tid] = aSqDiff;
		bSquareDiffs[tid] = bSqDiff;
	}
}
*/

__global__ void CudaGetPixelVals(int numPixels, int xPixels, int yPixels, thrust::device_ptr<float> pixelVals)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < numPixels)
	{
		int v = (int) tid / xPixels;
		int u = tid % xPixels;

		float4 val = tex2D(texRef, u, v);
		pixelVals[tid] = (val.x + val.y + val.z) / 3.0f;
	}
}


__global__ void CudaGetSquareDiffs(int numPixels, int xPixels, int yPixels, int aKernelRadius, thrust::device_ptr<float> aSquareDiffs, int bKernelRadius, thrust::device_ptr<float> bSquareDiffs, float meanPixelVal)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < numPixels)
	{
		int v = (int) tid / xPixels;
		int u = tid % xPixels;

		float4 centre = tex2D(texRef, u, v);
		float centreVal = (centre.x + centre.y + centre.z) / 3.0f;

		float aTop = 0.0f;
		float bTop = 0.0f;
		int aNum = 0;
		int bNum = 0;

		for (int i = u - bKernelRadius; i <= u + bKernelRadius; i++)
		{
			for (int j = v - bKernelRadius; j <= v + bKernelRadius; j++)
			{
				float dist2 = glm::distance2(glm::vec2(i, j), glm::vec2(u, v));

				if (dist2 > bKernelRadius * bKernelRadius)
					continue;

				float4 sample = tex2D(texRef, i, j);
				float sampleVal = (sample.x + sample.y + sample.z) / 3.0f;

				float top = (centreVal - meanPixelVal) * (sampleVal - meanPixelVal);

				bTop += top;
				bNum++;

				if (dist2 < aKernelRadius * aKernelRadius)
				{
					aTop += top;
					aNum++;
				}
			}
		}

		float bottom = (centreVal - meanPixelVal) * (centreVal - meanPixelVal);

		aSquareDiffs[tid] = aTop / (float)(aNum * bottom);
		bSquareDiffs[tid] = bTop / (float)(bNum * bottom);
	}
}


void ImageProcessor::GetAutoCorrelation()
{
	framebuffer->Unbind();

	thrust::fill(aSquareDiffs.begin(), aSquareDiffs.end(), 0.0f);
	thrust::fill(bSquareDiffs.begin(), bSquareDiffs.end(), 0.0f);

	HANDLE_ERROR( cudaGraphicsGLRegisterImage(&cudaResource, fbTex->ID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone) );
	HANDLE_ERROR( cudaGraphicsMapResources(1, &cudaResource) );

	cudaArray *cArray = 0;	
	HANDLE_ERROR( cudaGraphicsSubResourceGetMappedArray(&cArray, cudaResource, 0, 0) ); 
	HANDLE_ERROR( cudaBindTextureToArray(texRef, cArray) );

	CudaGetPixelVals<<<(numPixels + 255) / 256, 256>>>(numPixels, xPixels, yPixels, &pixelVals[0]);

	float meanPixelVal = thrust::reduce(pixelVals.begin(), pixelVals.end(), 0.0f, thrust::plus<float>());
	meanPixelVal /= (float)numPixels;


	CudaGetSquareDiffs<<<(numPixels + 255) / 256, 256>>>(numPixels, xPixels, yPixels, A_SQ_DIFF_RADIUS, &aSquareDiffs[0], B_SQ_DIFF_RADIUS, &bSquareDiffs[0], meanPixelVal);

	aResult = thrust::reduce(aSquareDiffs.begin(), aSquareDiffs.end(), 0.0f, thrust::plus<float>());
	bResult = thrust::reduce(bSquareDiffs.begin(), bSquareDiffs.end(), 0.0f, thrust::plus<float>());

//	std::cout << aResult << " - " << bResult << std::endl;

	HANDLE_ERROR( cudaUnbindTexture(texRef) );
	HANDLE_ERROR( cudaGraphicsUnmapResources(1, &cudaResource) );
	HANDLE_ERROR( cudaGraphicsUnregisterResource(cudaResource) );


//	std::string targetFileName = "dump.txt";
//	ofstream outStream(targetFileName, std::ios::app);
//
//	if (outStream.is_open())
//	{
//		outStream << aResult << "\t\t" << bResult << std::endl;
//		outStream.close();
//	}
}


__global__ void CudaWriteACToTexture(int numPixels, int xPixels, int yPixels, thrust::device_ptr<float> aSquareDiffs, thrust::device_ptr<float> bSquareDiffs, cudaSurfaceObject_t surface)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < numPixels)
	{
		int v = (int) tid / xPixels;
		int u = tid % xPixels;

		float val = bSquareDiffs[tid] - aSquareDiffs[tid];
//		glm::vec4 vec = glm::vec4(val, val, val, 1.0f);

		float4 vec;
		vec.x = val;
		vec.y = val;
		vec.z = val;
		vec.w = val;

//		vec.x = 1.0f;
//		vec.y = 1.0f;
//		vec.z = 1.0f;
//		vec.w = 1.0f;

	//	surf2Dread(&vec, surface, u*4*4, v);

//		if (v < 720)
			surf2Dwrite(vec, surface, u*4*4, v);
//		surf2Dwrite(vec, surface, 1*4*4, 2);
//		surf2Dwrite(vec, surface, 0, 1);
//		surf2Dwrite(vec, surface, 1279*4*4, 719);
	}
}

void ImageProcessor::WriteToTexture()
{
	HANDLE_ERROR( cudaGraphicsGLRegisterImage(&cudaResource, fbTex->ID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore) );
	HANDLE_ERROR( cudaGraphicsMapResources(1, &cudaResource) );

	cudaArray *cArray = 0;	
	HANDLE_ERROR( cudaGraphicsSubResourceGetMappedArray(&cArray, cudaResource, 0, 0) ); 

	cudaResourceDesc wdsc;
	wdsc.resType = cudaResourceTypeArray;
	wdsc.res.array.array = cArray;
	cudaSurfaceObject_t writeSurface;
	HANDLE_ERROR( cudaCreateSurfaceObject(&writeSurface, &wdsc) );

	CudaWriteACToTexture<<<(numPixels + 255) / 256, 256>>>(numPixels, xPixels, yPixels, &aSquareDiffs[0], &bSquareDiffs[0], writeSurface);

	cudaDestroySurfaceObject(writeSurface);

	HANDLE_ERROR( cudaGraphicsUnmapResources(1, &cudaResource) );
	HANDLE_ERROR( cudaGraphicsUnregisterResource(cudaResource) );
}




void ImageProcessor::WriteToImage(int currentTimestep)
{
	glBindTexture(GL_TEXTURE_2D, fbTex->ID);

	ILboolean success;
	ILuint imageID;
	ilInit();
//	std::vector<glm::vec4> pixelBuffer(numPixels);
	std::vector<unsigned char> pixelBuffer(numPixels * 4);

	std::string prefix;

	if (currentTimestep < 10)
		prefix = "00";
	else if (currentTimestep < 100)
		prefix = "0";
	else
		prefix = "";

	std::string imageName("../Images/" + std::to_string(currentTimestep) + "/Ep_0_00__Time_" + prefix + std::to_string(currentTimestep) + ".png");
	ilGenImages(1, &imageID);
	ilBindImage(imageID);

//	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, &pixelBuffer[0]);

	glReadPixels(0, 0, xPixels, yPixels, GL_RGBA, GL_UNSIGNED_BYTE, &pixelBuffer[0]);

	GLenum err = glGetError();

	if (err != GL_NO_ERROR)
	    printf("glError: %s\n", gluErrorString(err));

	success = ilTexImage(xPixels, yPixels, 0, 4, IL_RGBA, IL_UNSIGNED_BYTE, &pixelBuffer[0]);

	success = ilSave(IL_PNG, imageName.c_str());

	ilDeleteImages(1, &imageID);

	glBindTexture(GL_TEXTURE_2D, 0);
}


void ImageProcessor::End()
{
	framebuffer->Unbind();
	fbTex->Render();
}
