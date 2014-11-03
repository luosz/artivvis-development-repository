#include "VisibilityHist.h"

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

	glGenFramebuffers (1, &frameBuffer);
	glBindFramebuffer (GL_FRAMEBUFFER, frameBuffer);

	unsigned int rb = 0;
	glGenRenderbuffers (1, &rb);
	glBindRenderbuffer (GL_RENDERBUFFER, rb);
	glRenderbufferStorage (GL_RENDERBUFFER, GL_DEPTH_COMPONENT, screenWidth, screenHeight);
	glFramebufferRenderbuffer (GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb);

	glBindFramebuffer (GL_FRAMEBUFFER, 0);

	currentSlice = 0;
	numSlices = 256;
	opacityTex = GenerateSliceTexture();

	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, opacityTex, 0);

	visibilities.resize(256);
	numVis.resize(256);
	std::fill(visibilities.begin(), visibilities.end(), 0.0f);
	std::fill(numVis.begin(), numVis.end(), 0);
}

GLuint VisibilityHistogram::GenerateSliceTexture()
{
	GLuint tex;
	glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, xPixels, yPixels, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL); 

	glBindTexture(GL_TEXTURE_2D, 0);

	return tex;
}

void VisibilityHistogram::InitSlices()
{
	
//	slices.resize(numSlices);

//	for (int i=0; i<numSlices; i++)
//		slices[i] = GenerateSliceTexture();
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

void VisibilityHistogram::CalculateHistogram(VolumeDataset &volume, TransferFunction &transferFunction, ShaderManager shaderManager, Camera &camera)
{
	std::fill(visibilities.begin(), visibilities.end(), 0.0f);
	std::fill(numVis.begin(), numVis.end(), 0);

	GLuint shaderProgramID = shaderManager.UseShader(VisibilityShader);

	glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, opacityTex, 0);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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
	glBindTexture (GL_TEXTURE_1D, opacityTex);


	glm::vec3 closestCorner = FindClosestCorner(camera);
	glm::vec3 camDirection = camera.GetViewDirection();
	float dist = glm::dot(closestCorner - camera.position, camDirection);

	glm::vec3 rightVec = glm::normalize(glm::cross(camDirection, glm::vec3(0.0f, 1.0f, 0.0f)));
	glm::vec3 upVec = glm::normalize(glm::cross(camDirection, -rightVec));

	for (int i=0; i<numSlices; i++)
	{
		glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//		dist += currentSlice * 0.005f;
		dist += 0.005f;

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

		glReadPixels(0, 0, xPixels, yPixels, GL_RGBA, GL_FLOAT, pixelBuffer);


		for (int i=0; i<xPixels*yPixels; i++)
		{
			float scalar = pixelBuffer[i*4 + 2];

			if (scalar > 0.0f)
			{
				int bin = scalar * 255.0f;
				visibilities[bin] += pixelBuffer[i*4 + 0];
				numVis[bin]++;
			}
		}
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);


	for (int i=0; i<256; i++)
	{
		if (numVis[i] > 0.0f)
			visibilities[i] /= numVis[i];
	}
}



void VisibilityHistogram::DrawHistogram(ShaderManager shaderManager, Camera &camera)
{
	GLuint shaderProgramID = shaderManager.UseShader(SimpleShader);

	int uniformLoc;

	glm::mat4 model_mat = glm::mat4(1.0f);

	uniformLoc = glGetUniformLocation (shaderProgramID, "proj");
	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &camera.projMat[0][0]);

	uniformLoc = glGetUniformLocation (shaderProgramID, "view");
	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &camera.viewMat[0][0]);

	uniformLoc = glGetUniformLocation (shaderProgramID, "model");
	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &model_mat[0][0]);


	glBegin(GL_LINES);

	for (int i=0; i<256; i++)
	{
		glColor3f(1.0f, 0.0f, 0.0f);
		glVertex3f(i / 255.0f, 0.0f, 0.0f);
		glVertex3f(i / 255.0f, visibilities[i], 0.0f);
	}

	glEnd();
}