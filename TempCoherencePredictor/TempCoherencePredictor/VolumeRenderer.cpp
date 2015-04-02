#include "VolumeRenderer.h"

GLuint GenerateTexture2D(int xPixels, int yPixels)
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


GLuint ReadTexture()
{
	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	ILuint img;
	ilInit();
	ilGenImages(1, &img);
	ilBindImage(img);

	ILboolean success = ilLoadImage((const ILstring)"gv2.png");

	if(success)
	{
		success = ilConvertImage(IL_RGBA,IL_UNSIGNED_BYTE);
		glTexImage2D(GL_TEXTURE_2D, 0, ilGetInteger(IL_IMAGE_BPP), ilGetInteger(IL_IMAGE_WIDTH), ilGetInteger(IL_IMAGE_HEIGHT),0, ilGetInteger(IL_IMAGE_FORMAT), GL_UNSIGNED_BYTE, ilGetData());
	}

	ilDeleteImages(1, &img);

	return tex;
}

void VolumeRenderer::Init(int screenWidth, int screenHeight)
{
	glEnable(GL_DEPTH_TEST);
//	glDisable(GL_DEPTH_TEST);

	camera.Init(screenWidth, screenHeight);
	shaderManager.Init();
	volume.Init();


//	tempCoherence = new TempCoherence(screenWidth, screenHeight, volume);
	bruteForce = new BruteForce(volume);

	raycaster = new Raycaster(screenWidth, screenHeight, volume);

	transferFunction.Init(" ", volume);

	tester.Init(screenWidth, screenHeight, volume);


	writeToFile = false;
//	if (writeToFile)
//		fileWriter.Init(screenWidth, screenHeight);

	gridTexture = GenerateTexture2D(screenWidth, screenHeight);
	volumeTexture = GenerateTexture2D(screenWidth, screenHeight);
	framebuffer.Generate(screenWidth, screenHeight, gridTexture);

	oldTime = clock();

	imageTexture = ReadTexture();
}




void VolumeRenderer::Update()
{
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	camera.Update();
	
//	if (volume.timesteps > 1)
//	{
		clock_t currentTime = clock();
		float time = (currentTime - oldTime) / (float) CLOCKS_PER_SEC;

		if (time > volume.timePerFrame)
		{
			if (currentTimestep < volume.timesteps - 1)
				currentTimestep++;
			else
				currentTimestep = 0;

			oldTime = currentTime;

//			interpTex3D = tempCoherence->TemporalCoherence(volume, currentTimestep, transferFunction, shaderManager, camera);
			bruteTex3D = bruteForce->BruteForceCopy(volume, currentTimestep);

//			tester.Test(volume, transferFunction, shaderManager, camera, *raycaster, bruteTex3D, interpTex3D, currentTimestep);
			
//			if (writeToFile)
//				fileWriter.Write(currentTimestep, *tempCoherence, tester);

		}
//	}
//	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer.ID);	
//	
//	
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
//	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glEnable(GL_DEPTH_TEST);

	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, volumeTexture, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	GLuint shaderProgramID = shaderManager.UseShader(TFShader);
	raycaster->Raycast(transferFunction, shaderProgramID, camera, bruteTex3D);
//
//	glDisable(GL_DEPTH_TEST);

//	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gridTexture, 0);
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//	tempCoherence->DrawBoxes(volume, shaderManager, camera);
	
	glBindFramebuffer(GL_FRAMEBUFFER, 0);


	
	shaderProgramID = shaderManager.UseShader(TextureComboShader);
	
	glActiveTexture (GL_TEXTURE0);
	int texLoc = glGetUniformLocation(shaderProgramID,"volumeTex");
	glUniform1i(texLoc,0);
	glBindTexture (GL_TEXTURE_2D, volumeTexture);	


	glActiveTexture (GL_TEXTURE1);
	texLoc = glGetUniformLocation(shaderProgramID,"backgroundTex");
	glUniform1i(texLoc,1);
	glBindTexture (GL_TEXTURE_2D, imageTexture);	

	int texcoords_location = glGetAttribLocation (shaderProgramID, "vTexture");

	glBegin(GL_QUADS);
	glVertexAttrib2f(texcoords_location, 0.0f, 0.0f);
	glVertex2f(-1.0f, -1.0f);

	glVertexAttrib2f(texcoords_location, 1.0f, 0.0f);
	glVertex2f(1.0f, -1.0f);

	glVertexAttrib2f(texcoords_location, 1.0f, 1.0f);
	glVertex2f(1.0f, 1.0f);

	glVertexAttrib2f(texcoords_location, 0.0f, 1.0f);
	glVertex2f(-1.0f, 1.0f);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, 0);

//	fileWriter.WriteImage(currentTimestep);
//
//	ofstream outStream("ratio_1_00.txt", std::ios::app);
//
//	if (outStream.is_open())
//	{
//		outStream << (float)tempCoherence->numBlocksExtrapolated / (float)tempCoherence->numBlocks << std::endl;  
//		outStream.close();
//	}

	glutSwapBuffers();

	if (currentTimestep == 499)
			getchar();
}


