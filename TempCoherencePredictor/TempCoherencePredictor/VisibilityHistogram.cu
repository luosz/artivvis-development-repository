#include "VisibilityHistogram.h"

texture <float4, cudaTextureType2D, cudaReadModeElementType> texRef;

VisibilityHistogram::VisibilityHistogram(int screenWidth, int screenHeight)
{
	xPixels = screenWidth;
	yPixels = screenHeight;

	boxCorners.push_back(glm::vec3(1.0f, 1.0f, 1.0f));
	boxCorners.push_back(glm::vec3(1.0f, 1.0f, -1.0f));
	boxCorners.push_back(glm::vec3(1.0f, -1.0f, -1.0f));
	boxCorners.push_back(glm::vec3(1.0f, -1.0f, 1.0f));
	boxCorners.push_back(glm::vec3(-1.0f, 1.0f, 1.0f));
	boxCorners.push_back(glm::vec3(-1.0f, 1.0f, -1.0f));
	boxCorners.push_back(glm::vec3(-1.0f, -1.0f, -1.0f));
	boxCorners.push_back(glm::vec3(-1.0f, -1.0f, 1.0f));

	// Generate two textures for alternating read and write on framebuffer
	opacityTex1 = GenerateSliceTexture();
	opacityTex2 = GenerateSliceTexture();
	
	frameBuffer.Generate(xPixels, yPixels, opacityTex1);

	currentSlice = 0;
	numSlices = 512;
	numBins = 256;
	
	values.resize(numBins);
	std::fill(values.begin(), values.end(), 0.0f);

	// Allocate memory on GPU
	HANDLE_ERROR( cudaMalloc((void**)&cudaHistBins, 256 * sizeof(float)) );
	HANDLE_ERROR( cudaMalloc((void**)&cudaNumInBin, 256 * sizeof(int)) );

	grabFrustum = false;
	frustumExtent = 5;
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

// Finds closest point from the camera the volume
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

// Finds farthest point from the camera the volume
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

// Uses atomic adds to accumulate visibility values for respective intensity values, read directly from texture memory
__global__ void CudaEvaluate(int xPixels, int yPixels, float *histBins, int *numInBin)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < xPixels * yPixels)
	{
		int v = (int) tid / xPixels;
		int u = tid % xPixels;

		float4 color = tex2D(texRef, u, v);
		float scalar = color.z;

		if (scalar > 0.0f)
		{
			int bin = scalar * 255.0f;

	//		if (u == 400 && v == 400)
	//			printf("%d, %f\n", bin, color.x);

			atomicAdd(&(histBins[bin]), (float)color.x);
			atomicAdd(&(numInBin[bin]), (int)1);
		}
	}
}

// Find average visbility for each bin
__global__ void CudaNormalize(int numBins, float *histBins, int *numInBin)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < numBins)
	{
		if (numInBin[tid] > 0)
		{
		//	if (tid == 44)
		//		printf("%d, %f\n", numInBin[tid], histBins[tid]);
			histBins[tid] = histBins[tid] / numInBin[tid];
		}
	}
}


__global__ void CudaGrabFrustum(int numFrustumPixels, int frustumExtent, int mousePosX, int mousePosY, float *histBins, int *numInBin)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < numFrustumPixels)
	{
		int width = (frustumExtent * 2) + 1;

		int bottomCornerX = mousePosX - frustumExtent;
		int bottomCornerY = mousePosY - frustumExtent;

		int v = ((int) tid / width) + bottomCornerY;
		int u = (tid % width) + bottomCornerX;

//		printf("%d: %d, %d - %d, %d\n", tid, u, v, mousePosX, mousePosY);

		float4 color = tex2D(texRef, u, v);
		float scalar = color.z;

		if (scalar > 0.0f)
		{
			int bin = scalar * 255.0f;

			atomicAdd(&(histBins[bin]), (float)color.x);
			atomicAdd(&(numInBin[bin]), (int)1);
		}
	}
}



// Calculate visibility histogram
void VisibilityHistogram::Update(int currentTimestep, VolumeDataset &volume, GLuint tex3D, GLuint &tfTexture, ShaderManager &shaderManager, Camera &camera)
{
	// Set all values for GPU memory to zero
	HANDLE_ERROR( cudaMemset(cudaHistBins, 0, 256 * sizeof(float)) );
	HANDLE_ERROR( cudaMemset(cudaNumInBin, 0, 256 * sizeof(int)) );

	// Bind visibility shader and framebuffer with a write texture attached
	GLuint shaderProgramID = shaderManager.UseShader(VisibilityShader);

	glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer.ID);
	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, opacityTex2, 0);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	// Pass in uniforms
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
	glBindTexture (GL_TEXTURE_3D, tex3D);

	glActiveTexture (GL_TEXTURE1);
	uniformLoc = glGetUniformLocation(shaderProgramID,"transferFunc");
	glUniform1i(uniformLoc,1);
	glBindTexture (GL_TEXTURE_1D, tfTexture);


	// Find closest point on volume to camera to start slicing there
	glm::vec3 closestCorner = FindClosestCorner(camera);
	glm::vec3 farthestCorner = FindFarthestCorner(camera);

	glm::vec3 camDirection = camera.GetViewDirection();
	float dist = glm::dot(closestCorner - camera.position, camDirection);

	// Find perpendicular vectors to calculate slice corners
	glm::vec3 rightVec = glm::normalize(glm::cross(camDirection, glm::vec3(0.0f, 1.0f, 0.0f)));
	glm::vec3 upVec = glm::normalize(glm::cross(camDirection, -rightVec));

	float farDist = glm::dot(farthestCorner - camera.position, camDirection);
	float sliceGap = (farDist - dist) / (float)numSlices; 

	GLuint readTex, writeTex;

	for (int i=0; i<numSlices; i++)
	{
		// Alternate read/write textures
		if (i % 2 == 0)
		{
			writeTex = opacityTex1;
			readTex = opacityTex2;
		}
		else
		{
			writeTex = opacityTex2;
			readTex = opacityTex1;
		}

		glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, writeTex, 0);
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glActiveTexture (GL_TEXTURE2);
		glBindTexture (GL_TEXTURE_2D, readTex);
		uniformLoc = glGetUniformLocation(shaderProgramID,"opacityTex");
		glUniform1i(uniformLoc,2);

		dist += 0.005f;

		float extent = dist * glm::tan((camera.FoV / 2.0f) * (glm::pi<float>()/180.0f));
		glm::vec3 centre = camera.position + (camDirection * dist);

		// Calculate corners of slice using perpendicular vectors, my slice takes up entire window
		glm::vec3 topLeft = centre + (extent * upVec) - (extent * rightVec);
		glm::vec3 topRight = centre + (extent * upVec) + (extent * rightVec);
		glm::vec3 bottomLeft = centre - (extent * upVec) - (extent * rightVec);
		glm::vec3 bottomRight = centre - (extent * upVec) + (extent * rightVec);

		int texcoords_location = glGetAttribLocation (shaderProgramID, "vTexture");

		// Initiated the screen aligned quad
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


		// Map cuda memory to texture memory, big time saver
		HANDLE_ERROR( cudaGraphicsGLRegisterImage(&resource, writeTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone) );
		HANDLE_ERROR( cudaGraphicsMapResources(1, &resource) );
		cudaArray *arry = 0;	
		HANDLE_ERROR( cudaGraphicsSubResourceGetMappedArray(&arry, resource, 0, 0) ); 
		HANDLE_ERROR( cudaBindTextureToArray(texRef, arry) );

		if (!grabFrustum)
		{
			// Launch CUDA kernel to accumulate visbility values in parallel
			int numPixels = xPixels * yPixels;
			CudaEvaluate<<<(numPixels + 255) / 256, 256>>>(xPixels, yPixels, cudaHistBins, cudaNumInBin);
		}
		else
		{
			int numFrustumPixels = ((frustumExtent*2) + 1) * ((frustumExtent*2) + 1);
			CudaGrabFrustum <<<(numFrustumPixels + 255) / 256, 256>>> (numFrustumPixels, frustumExtent, mousePosX, 800 - mousePosY, cudaHistBins, cudaNumInBin);
		}

		// Unbind and unmap, must be done before OpenGL uses texture memory again
		HANDLE_ERROR( cudaUnbindTexture(texRef) );
		HANDLE_ERROR( cudaGraphicsUnmapResources(1, &resource) );
		HANDLE_ERROR( cudaGraphicsUnregisterResource(resource) );
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// Average visbilities
	CudaNormalize<<< (numBins + 255) / 256, 256>>>(256, cudaHistBins, cudaNumInBin);

	// Copy visibility info back to CPU memory for ease of access
	HANDLE_ERROR( cudaMemcpy(&values[0], cudaHistBins, 256 * sizeof(float), cudaMemcpyDeviceToHost) );
}


// Draw visibility histogram
void VisibilityHistogram::DrawHistogram(ShaderManager &shaderManager, Camera &camera)
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
		glVertex3f(i / 255.0f, values[i], 0.0f);
	}

	glEnd();
}



