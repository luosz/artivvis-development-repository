#include "VisibilityHist.h"

texture <float4, cudaTextureType2D, cudaReadModeElementType> texRef;

void VisibilityHistogram::Init(int screenWidth, int screenHeight)
{
	xPixels = screenWidth;
	yPixels = screenHeight;

	pixelBuffer = new float[screenWidth * screenHeight * 4];

	boxCorners.push_back(glm::vec3(1.0f, 1.0f, 1.0f));
	boxCorners.push_back(glm::vec3(1.0f, 1.0f, -1.0f));
	boxCorners.push_back(glm::vec3(1.0f, -1.0f, -1.0f));
	boxCorners.push_back(glm::vec3(1.0f, -1.0f, 1.0f));
	boxCorners.push_back(glm::vec3(-1.0f, 1.0f, 1.0f));
	boxCorners.push_back(glm::vec3(-1.0f, 1.0f, -1.0f));
	boxCorners.push_back(glm::vec3(-1.0f, -1.0f, -1.0f));
	boxCorners.push_back(glm::vec3(-1.0f, -1.0f, 1.0f));

	opacityTex = GenerateSliceTexture();

	

	glGenFramebuffers (1, &frameBuffer);
	glBindFramebuffer (GL_FRAMEBUFFER, frameBuffer);

	unsigned int rb = 0;
	glGenRenderbuffers (1, &rb);
	glBindRenderbuffer (GL_RENDERBUFFER, rb);
	glRenderbufferStorage (GL_RENDERBUFFER, GL_DEPTH_COMPONENT, screenWidth, screenHeight);
	glFramebufferRenderbuffer (GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb);

	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, opacityTex, 0);

	glBindFramebuffer (GL_FRAMEBUFFER, 0);

	currentSlice = 0;
	numSlices = 256;
	numBins = 256;
	

	visibilities.resize(numBins);
	numVis.resize(numBins);
	std::fill(visibilities.begin(), visibilities.end(), 0.0f);
	std::fill(numVis.begin(), numVis.end(), 0);

	HANDLE_ERROR( cudaMalloc((void**)&cudaHistBins, 256 * sizeof(float)) );
	HANDLE_ERROR( cudaMalloc((void**)&cudaNumInBin, 256 * sizeof(int)) );

//	thrust::device_vector<float> blah(100);
//	cudaHistBins.resize(256);
//	cudaNumInBin.resize(256);

	HANDLE_ERROR( cudaGraphicsGLRegisterImage(&resource, opacityTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone) );

}

GLuint VisibilityHistogram::GenerateSliceTexture()
{
	GLuint tex;
	glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, xPixels, yPixels, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL); 

	glBindTexture(GL_TEXTURE_2D, 0);

	return tex;
}


glm::vec3 VisibilityHistogram::FindClosestCorner(Camera &camera)
{
	int minPoint = 0;
	float minDist = 10000.0f;
	float newDist;

	for (int i=0; i<boxCorners.size(); i++)
	{
		newDist = glm::distance2(camera.position, boxCorners[i]);
		
		if (newDist < minDist)
		{
			minDist = newDist;
			minPoint = i;
		}
	}

	return boxCorners[minPoint];
}

glm::vec3 VisibilityHistogram::FindFarthestCorner(Camera &camera)
{
	int maxPoint = 0;
	float maxDist = 0.0f;
	float newDist;

	for (int i=0; i<boxCorners.size(); i++)
	{
		newDist = glm::distance2(camera.position, boxCorners[i]);
		
		if (newDist > maxDist)
		{
			maxDist = newDist;
			maxPoint = i;
		}
	}

	return boxCorners[maxPoint];
}

__global__ void CudaEvaluate(int xPixels, int yPixels, float *histBins, int *numInBin)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < xPixels * yPixels)
	{
		int v = (int) tid / yPixels;
		int u = tid % yPixels;

		float4 color = tex2D(texRef, u, v);
		float scalar = color.z;

		if (scalar > 0.0f)
		{
//			printf("%d, %d, %d\n", tid, u, v);
			int bin = scalar * 255.0f;
//			if (scalar > 0.0f)
//			if (u == 400 && v == 400)
//				printf("%d: %f, %f, %f %f\n", bin, color.x, color.y, color.z, color.w);
			atomicAdd(&(histBins[bin]), (float)color.x);
			atomicAdd(&(numInBin[bin]), (int)1);
		}
	}
}

__global__ void CudaCheck(int xPixels, int yPixels)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < xPixels * yPixels)
	{
		int v = (int) tid / yPixels;
		int u = tid % yPixels;

		float4 color = tex2D(texRef, u, v);
		float scalar = color.z;

		if (color.x > 0.0f)
		{
//			printf("%d, %d, %d\n", tid, u, v);
			int bin = scalar * 255.0f;
//			if (scalar > 0.8f)
//			if (u == 400 && v == 400)
				printf("%d: %f, %f, %f %f\n", bin, color.x, color.y, color.z, color.w);
		}
	}
}

__global__ void CudaNormalize(int numBins, float *histBins, int *numInBin)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < numBins)
	{
		if (numInBin[tid] > 0)
		{
//			printf("%d, %f, %d\n", tid, histBins[tid], numInBin[tid]);
			histBins[tid] = histBins[tid] / numInBin[tid];
		}
	}
}

void VisibilityHistogram::CalculateHistogram(VolumeDataset &volume, TransferFunction &transferFunction, ShaderManager shaderManager, Camera &camera)
{
//	std::fill(visibilities.begin(), visibilities.end(), 0.0f);
//	std::fill(numVis.begin(), numVis.end(), 0);

	HANDLE_ERROR( cudaMemset(cudaHistBins, 0, 256 * sizeof(float)) );
	HANDLE_ERROR( cudaMemset(cudaNumInBin, 0, 256 * sizeof(int)) );

//	thrust::fill(cudaHistBins.begin(), cudaHistBins.end(), 0.0f);
//	thrust::fill(cudaNumInBin.begin(), cudaNumInBin.end(), 0);

	GLuint shaderProgramID = shaderManager.UseShader(VisibilityShader);

	glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
//	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, opacityTex, 0);
	

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

	glActiveTexture (GL_TEXTURE1);
	uniformLoc = glGetUniformLocation(shaderProgramID,"transferFunc");
	glUniform1i(uniformLoc,1);
	glBindTexture (GL_TEXTURE_1D, transferFunction.tfTexture);

	glActiveTexture (GL_TEXTURE2);
	uniformLoc = glGetUniformLocation(shaderProgramID,"opacityTex");
	glUniform1i(uniformLoc,2);
	glBindTexture (GL_TEXTURE_2D, opacityTex);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	glm::vec3 closestCorner = FindClosestCorner(camera);
	glm::vec3 farthestCorner = FindFarthestCorner(camera);

	glm::vec3 camDirection = camera.GetViewDirection();
	float dist = glm::dot(closestCorner - camera.position, camDirection);

	glm::vec3 rightVec = glm::normalize(glm::cross(camDirection, glm::vec3(0.0f, 1.0f, 0.0f)));
	glm::vec3 upVec = glm::normalize(glm::cross(camDirection, -rightVec));

	float farDist = glm::dot(farthestCorner - camera.position, camDirection);
	float sliceGap = (farDist - dist) / (float)numSlices; 

	for (int i=0; i<numSlices; i++)
	{
//		glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//		dist += currentSlice * 0.005f;
		dist += 0.01f;
//		dist += sliceGap;

		float extent = dist * glm::tan((camera.FoV / 2.0f) * (glm::pi<float>()/180.0f));
		glm::vec3 centre = camera.position + (camDirection * dist);

		glm::vec3 topLeft = centre + (extent * upVec) - (extent * rightVec);
		glm::vec3 topRight = centre + (extent * upVec) + (extent * rightVec);
		glm::vec3 bottomLeft = centre - (extent * upVec) - (extent * rightVec);
		glm::vec3 bottomRight = centre - (extent * upVec) + (extent * rightVec);

		int texcoords_location = glGetAttribLocation (shaderProgramID, "vTexture");

		glBegin(GL_QUADS);
		glVertexAttrib2f(texcoords_location, 1.0f, 1.0f);
		glVertex3f(topRight.x, topRight.y, topRight.z);

		glVertexAttrib2f(texcoords_location, 0.0f, 1.0f);
		glVertex3f(topLeft.x, topLeft.y, topLeft.z);

		glVertexAttrib2f(texcoords_location, 0.0f, 0.0f);
		glVertex3f(bottomLeft.x, bottomLeft.y, bottomLeft.z);

		glVertexAttrib2f(texcoords_location, 1.0f, 0.0f);
		glVertex3f(bottomRight.x, bottomRight.y, bottomRight.z);
		glEnd();

//		glReadPixels(0, 0, xPixels, yPixels, GL_RGBA, GL_FLOAT, pixelBuffer);

//		glBindTexture(GL_TEXTURE_2D, 0);


		HANDLE_ERROR( cudaGraphicsMapResources(1, &resource) );
		cudaArray *arry = 0;
		
		HANDLE_ERROR( cudaGraphicsSubResourceGetMappedArray(&arry, resource, 0, 0) ); 


		HANDLE_ERROR( cudaBindTextureToArray(texRef, arry) );

		int numPixels = xPixels * yPixels;
		CudaEvaluate<<<(numPixels + 255) / 256, 256>>>(xPixels, yPixels, cudaHistBins, cudaNumInBin);

		HANDLE_ERROR( cudaUnbindTexture(texRef) );
		HANDLE_ERROR( cudaGraphicsUnmapResources(1, &resource) );

		cudaDeviceSynchronize();
//		int a =0;
		


//		for (int j=0; j<xPixels*yPixels; j++)
//		{
//			float scalar = pixelBuffer[j*4 + 2];
//
//			if (scalar > 0.0f)
//			{
//				int bin = scalar * 255.0f;
//				visibilities[bin] += pixelBuffer[j*4 + 0];
//				numVis[bin]++;
//			}
//		}
	}

//	glBindTexture (GL_TEXTURE_3D, 0);
//	glBindTexture (GL_TEXTURE_2D, 0);
//	glBindTexture (GL_TEXTURE_1D, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	CudaNormalize<<< (numBins + 255) / 256, 256>>>(256, cudaHistBins, cudaNumInBin);

	HANDLE_ERROR( cudaMemcpy(&visibilities[0], cudaHistBins, 256 * sizeof(float), cudaMemcpyDeviceToHost) );

//	for (int i=0; i<256; i++)
//	{
//		if (numVis[i] > 0.0f)
//			visibilities[i] /= numVis[i];
//	}
}



void VisibilityHistogram::DrawHistogram(ShaderManager shaderManager, Camera &camera)
{
	GLuint shaderProgramID = shaderManager.UseShader(SimpleShader);

	int uniformLoc;

	glm::mat4 model_mat = glm::mat4(1.0f);

	uniformLoc = glGetUniformLocation (shaderProgramID, "proj");
	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &camera.projMat[0][0]);

	uniformLoc = glGetUniformLocation (shaderProgramID, "view");
	glm::mat4 tempView = glm::lookAt(glm::vec3(0.5f, 0.5f, 2.0f), glm::vec3(0.5f, 0.5f, 0.0f), glm::vec3(0.0f,1.0f,0.0f));
	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &tempView[0][0]);

	uniformLoc = glGetUniformLocation (shaderProgramID, "model");
	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &model_mat[0][0]);


	glBegin(GL_LINES);

	glColor3f(1.0f, 1.0f, 1.0f);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 1.0f, 0.0f);

	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(1.0f, 0.0f, 0.0f);

	glColor3f(1.0f, 0.0f, 0.0f);
	for (int i=0; i<256; i++)
	{
		glVertex3f(i / 255.0f, 0.0f, 0.0f);
		glVertex3f(i / 255.0f, visibilities[i], 0.0f);
	}

	glEnd();
}