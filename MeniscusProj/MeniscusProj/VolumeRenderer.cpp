#include "VolumeRenderer.h"

void VolumeRenderer::Init(int screenWidth, int screenHeight)
{
	glEnable(GL_DEPTH_TEST);

	camera.Init(screenWidth, screenHeight);
	shaderManager.Init();
	volume.Init();

	renderer = new OpenGLRenderer(screenWidth, screenHeight, volume, shaderManager, camera);

	currentTimestep = 0;
	focused = false;
	removed = false;
	spherePoint = glm::vec3(0.0f);
	sphereRadius = 0.1f;
}


void VolumeRenderer::Update()
{
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	

	camera.Update();
	
	// If dataset is time variant advance time and update GPU's texture
	if (volume.timesteps > 1)
	{
		clock_t currentTime = clock();
		float time = (currentTime - oldTime) / (float) CLOCKS_PER_SEC;

		if (time > volume.timePerFrame)
		{
			if (currentTimestep < volume.timesteps - 2)
				currentTimestep++;
			else
				currentTimestep = 0;

			oldTime = currentTime;

			renderer->UpdateTexture(currentTimestep, volume);
		}	
	}

	FindRemoved();

	renderer->Draw(volume, shaderManager, camera, clipPlane, focused, removed, sphereRadius, spherePoint);
	DrawBox();
	DrawMyText();

	glutSwapBuffers();
}


void VolumeRenderer::FindRemoved()
{
	if (spherePoint == glm::vec3(0.0f))
		return;

	numInFocused = 0;
	numRemoved = 0;
	int ID = 0;

	for (int z=0; z<volume.zRes; z++)
		for (int y=0; y<volume.yRes; y++)
			for (int x=0; x<volume.xRes; x++)
			{
				float voxVal = (float)volume.memblock3D[ID] / 255.0f;

				if (glm::abs(voxVal - renderer->tfBandPos) < renderer->tfBandWidth)
				{
					numInFocused++;

					glm::vec3 normalizedPos;

					normalizedPos.x = ((x / (float)volume.xRes) * 2.0f) - 1.0f;
					normalizedPos.y = ((y / (float)volume.yRes) * 2.0f) - 1.0f;
					normalizedPos.z = ((z / (float)volume.zRes) * 2.0f) - 1.0f;

					if (glm::distance(normalizedPos, spherePoint) < sphereRadius)
						numRemoved++;
				}
				ID++;
			}
}

void VolumeRenderer::DrawMyText()
{
	glActiveTexture (GL_TEXTURE0);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_TEXTURE_3D);
	
//	std::string tfWidthString = "TF Band Width: " + std::to_string(renderer->tfBandWidth);
//	std::string tfPosString = "TF Band Pos: " + std::to_string(renderer->tfBandPos);

	std::ostringstream tfWidthString;
	tfWidthString << "tfWidth: " << std::setprecision(2) << renderer->tfBandWidth;

	std::ostringstream tfPosString;
	tfPosString << "tfPos: " << std::setprecision(2) << renderer->tfBandPos;

	glUseProgram(0);
	glColor3f(1.0f, 1.0f, 1.0f);

	glRasterPos2f(-0.95f, 0.9f);
	glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)tfWidthString.str().c_str());

	
	glRasterPos2f(-0.95f, 0.8f);
	glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)tfPosString.str().c_str());	


	if (spherePoint != glm::vec3(0.0f))
	{
		std::ostringstream removedString;
		removedString << "% removed: " << std::setprecision(2) << ((float)numRemoved / (float)numInFocused) * 100.0f;

		glRasterPos2f(-0.95f, 0.7f);
		glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)removedString.str().c_str());
	}

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_3D);
}


void VolumeRenderer::DrawBox()
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glDisable(GL_DEPTH_TEST);

	GLuint shaderProgramID = shaderManager.UseShader(SimpleShader);


	GLint uniformLoc;

	glm::mat4 model_mat = glm::mat4(1.0f);

	uniformLoc = glGetUniformLocation (shaderProgramID, "proj");
	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &camera.projMat[0][0]);

	uniformLoc = glGetUniformLocation (shaderProgramID, "view");
	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &camera.viewMat[0][0]);

	uniformLoc = glGetUniformLocation (shaderProgramID, "model");
	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &model_mat[0][0]);

	glPointSize(4.0f);
	glColor3f(1.0f, 0.0f, 0.0f);
	glBegin(GL_POINTS);

	for (int i=0; i<boxPoints.size(); i++)
		glVertex3f(boxPoints[i].x, boxPoints[i].y, boxPoints[i].z);

	glEnd();


	if (boxPoints.size() >= 4)
	{
		glBegin(GL_QUADS);

		glVertex3f(boxPoints[0].x, boxPoints[0].y, boxPoints[0].z);
		glVertex3f(boxPoints[2].x, boxPoints[2].y, boxPoints[2].z);
		glVertex3f(boxPoints[1].x, boxPoints[1].y, boxPoints[1].z);
		glVertex3f(boxPoints[3].x, boxPoints[3].y, boxPoints[3].z);

		if (boxPoints.size() == 8)
		{
			glVertex3f(boxPoints[4].x, boxPoints[4].y, boxPoints[4].z);
			glVertex3f(boxPoints[6].x, boxPoints[6].y, boxPoints[6].z);
			glVertex3f(boxPoints[5].x, boxPoints[5].y, boxPoints[5].z);
			glVertex3f(boxPoints[7].x, boxPoints[7].y, boxPoints[7].z);

			glVertex3f(boxPoints[1].x, boxPoints[1].y, boxPoints[1].z);
			glVertex3f(boxPoints[2].x, boxPoints[2].y, boxPoints[2].z);
			glVertex3f(boxPoints[6].x, boxPoints[6].y, boxPoints[6].z);
			glVertex3f(boxPoints[5].x, boxPoints[5].y, boxPoints[5].z);


			glVertex3f(boxPoints[0].x, boxPoints[0].y, boxPoints[0].z);
			glVertex3f(boxPoints[3].x, boxPoints[3].y, boxPoints[3].z);
			glVertex3f(boxPoints[7].x, boxPoints[7].y, boxPoints[7].z);
			glVertex3f(boxPoints[4].x, boxPoints[4].y, boxPoints[4].z);
		}

		glEnd();
	}

	glEnable(GL_DEPTH_TEST);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}


void VolumeRenderer::AddBoxPoint(int xMousePos, int yMousePos)
{
	glm::vec3 intersersection = clipPlane.Intersect(xMousePos, yMousePos, camera);

	if (focused)
	{
		spherePoint = intersersection;
	}
	else
	{
		if (boxPoints.size() >= 8)
		{
			return;
		}
		else if (boxPoints.size() == 4)
		{
			if (clipPlane.normal.x == 1.0f)
			{
				boxPoints.push_back(glm::vec3(intersersection.x, boxPoints[0].y, boxPoints[0].z));
				boxPoints.push_back(glm::vec3(intersersection.x, boxPoints[1].y, boxPoints[1].z));
				boxPoints.push_back(glm::vec3(intersersection.x, boxPoints[2].y, boxPoints[2].z));
				boxPoints.push_back(glm::vec3(intersersection.x, boxPoints[3].y, boxPoints[3].z));
			}
			else if (clipPlane.normal.y == 1.0f)
			{
				boxPoints.push_back(glm::vec3(boxPoints[0].x, intersersection.y, boxPoints[0].z));
				boxPoints.push_back(glm::vec3(boxPoints[1].x, intersersection.y, boxPoints[1].z));
				boxPoints.push_back(glm::vec3(boxPoints[2].x, intersersection.y, boxPoints[2].z));
				boxPoints.push_back(glm::vec3(boxPoints[3].x, intersersection.y, boxPoints[3].z));
			}
			else if (clipPlane.normal.z == 1.0f)
			{
				boxPoints.push_back(glm::vec3(boxPoints[0].x, boxPoints[0].y, intersersection.z));
				boxPoints.push_back(glm::vec3(boxPoints[1].x, boxPoints[1].y, intersersection.z));
				boxPoints.push_back(glm::vec3(boxPoints[2].x, boxPoints[2].y, intersersection.z));
				boxPoints.push_back(glm::vec3(boxPoints[3].x, boxPoints[3].y, intersersection.z));
			}
		}
		else
		{
			boxPoints.push_back(intersersection);

			if (boxPoints.size() == 2)
			{
				if (clipPlane.normal.x == 1.0f)
				{
					boxPoints.push_back(glm::vec3(intersersection.x, boxPoints[0].y, boxPoints[1].z));
					boxPoints.push_back(glm::vec3(intersersection.x, boxPoints[1].y, boxPoints[0].z));
				}
				else if (clipPlane.normal.y == 1.0f)
				{
					boxPoints.push_back(glm::vec3(boxPoints[0].x, intersersection.y, boxPoints[1].z));
					boxPoints.push_back(glm::vec3(boxPoints[1].x, intersersection.y, boxPoints[0].z));
				}
				else if (clipPlane.normal.z == 1.0f)
				{
					boxPoints.push_back(glm::vec3(boxPoints[0].x, boxPoints[1].y, intersersection.z));
					boxPoints.push_back(glm::vec3(boxPoints[1].x, boxPoints[0].y, intersersection.z));
				}
			}
		}
	}
}




void VolumeRenderer::FocusVolume()
{
//	delete tempVol;
	tempVol = volume.memblock3D;
	tempXRes = volume.xRes;
	tempYRes = volume.yRes;
	tempZRes = volume.zRes;

	glm::vec3 minPoint = boxPoints[0];
	glm::vec3 maxPoint = boxPoints[0];

	for (int i=1; i<boxPoints.size(); i++)
	{
		minPoint.x = glm::min(minPoint.x, boxPoints[i].x);
		minPoint.y = glm::min(minPoint.y, boxPoints[i].y);
		minPoint.z = glm::min(minPoint.z, boxPoints[i].z);

		maxPoint.x = glm::max(maxPoint.x, boxPoints[i].x);
		maxPoint.y = glm::max(maxPoint.y, boxPoints[i].y);
		maxPoint.z = glm::max(maxPoint.z, boxPoints[i].z);
	}

	glm::ivec3 minVox, maxVox;

	minVox.x = (minPoint.x + 1.0f) / (2.0f / (float)volume.xRes);
	minVox.y = (minPoint.y + 1.0f) / (2.0f / (float)volume.yRes);
	minVox.z = (minPoint.z + 1.0f) / (2.0f / (float)volume.zRes);

	maxVox.x = (maxPoint.x + 1.0f) / (2.0f / (float)volume.xRes);
	maxVox.y = (maxPoint.y + 1.0f) / (2.0f / (float)volume.yRes);
	maxVox.z = (maxPoint.z + 1.0f) / (2.0f / (float)volume.zRes);

	int newXRes = maxVox.x - minVox.x;
	int newYRes = maxVox.y - minVox.y;
	int newZRes = maxVox.z - minVox.z;

	int newSize = newXRes * newYRes * newZRes * volume.bytesPerElement;
	volume.memblock3D = new GLubyte[newSize];

	for (int z=0; z<newZRes; z++)
		for (int y=0; y<newYRes; y++)
			for (int x=0; x<newXRes; x++)
			{
				int oldID = (x + minVox.x) + ((y + minVox.y) * tempXRes) + ((z + minVox.z) * tempXRes * tempYRes);
				int newID = x + (y * newXRes) + (z * newXRes * newYRes);

				if (volume.bytesPerElement == 1)
					volume.memblock3D[newID] = tempVol[oldID];
				else if (volume.bytesPerElement == 2)
					*(unsigned short*)(&volume.memblock3D[newID * 2]) = *(unsigned short*)(&tempVol[oldID * 2]);
			}

	volume.xRes = newXRes;
	volume.yRes = newYRes;
	volume.zRes = newZRes;

	renderer->currTexture3D = renderer->GenerateTexture(volume);

	boxPoints.clear();
	focused = true;
	clipPlane.point = clipPlane.normal;
}


void VolumeRenderer::Reset()
{
	volume.xRes	= tempXRes;
	volume.yRes	= tempYRes;
	volume.zRes	= tempZRes;
	
	delete volume.memblock3D;
	volume.memblock3D = tempVol;

	renderer->currTexture3D = renderer->GenerateTexture(volume);

	boxPoints.clear();
	focused = false;
	removed = false;
	spherePoint = glm::vec3(0.0f);
	sphereRadius = 0.1f;
}