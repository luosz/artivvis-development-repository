#include "GPUContours.h"

void GPUContours::Init(int screenWidth, int screenHeight, VolumeDataset &volume)
{
	numPixelsLower = 100;
	suggestiveContourThreshold = 0.05f;
	kernelRadius = 10;

	glGenFramebuffers (1, &frameBuffer);
	glBindFramebuffer (GL_FRAMEBUFFER, frameBuffer);

	unsigned int rb = 0;
	glGenRenderbuffers (1, &rb);
	glBindRenderbuffer (GL_RENDERBUFFER, rb);
	glRenderbufferStorage (GL_RENDERBUFFER, GL_DEPTH_COMPONENT, screenWidth, screenHeight);
	glFramebufferRenderbuffer (GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb);

	glBindFramebuffer (GL_FRAMEBUFFER, 0);

	glGenTextures(1, &diffuseTexture);
	glBindTexture (GL_TEXTURE_2D, diffuseTexture);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA32F, screenWidth, screenHeight, 0, GL_RGBA, GL_FLOAT, NULL);

	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, diffuseTexture, 0);


	glGenTextures(1, &opacityTexture);
	glBindTexture (GL_TEXTURE_2D, opacityTexture);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA32F, screenWidth, screenHeight, 0, GL_RGBA, GL_FLOAT, NULL);

	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, opacityTexture, 0);
}


void GPUContours::DrawContours(VolumeDataset &volume, Camera &camera, ShaderManager &shaderManager, Raycaster &raycaster)
{
//	int uniformLocation;
//	GLuint shaderProgramID = shaderManager.UseShader(DiffuseShader);
//
//	glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
//	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, diffuseTexture, 0);
//	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
//	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//	
//	
//	raycaster.Raycast(volume, shaderProgramID, camera);
//
//	
//	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, opacityTexture, 0);
//	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
//	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//	
//	
//	
//	shaderProgramID = shaderManager.UseShader(OpacityShader);
//	raycaster.Raycast(volume, shaderProgramID, camera);
//	
//
//	
//	glBindFramebuffer (GL_FRAMEBUFFER, 0);
//	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
//	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//	
//	
//
//	shaderProgramID = shaderManager.UseShader(ContourShader);
//
//	glActiveTexture (GL_TEXTURE0);
//	uniformLocation = glGetUniformLocation(shaderProgramID,"diffuseImage");
//	glUniform1i(uniformLocation, 0);
//	glBindTexture (GL_TEXTURE_2D, diffuseTexture);
//
//	
//	glActiveTexture (GL_TEXTURE1);
//	uniformLocation = glGetUniformLocation(shaderProgramID,"opacityImage");
//	glUniform1i(uniformLocation, 1);
//	glBindTexture (GL_TEXTURE_2D, opacityTexture);
//	
//
//	uniformLocation = glGetUniformLocation(shaderProgramID, "numPixelsLower");
//	glUniform1i(uniformLocation, numPixelsLower);
//
//	uniformLocation = glGetUniformLocation(shaderProgramID, "suggestiveContourThreshold");
//	glUniform1f(uniformLocation, suggestiveContourThreshold);
//
//	uniformLocation = glGetUniformLocation(shaderProgramID, "kernelRadius");
//	glUniform1i(uniformLocation, kernelRadius);
//
//	int texcoords_location = glGetAttribLocation (shaderProgramID, "vTexture");
//	glBegin(GL_QUADS);
//	glVertexAttrib2f(texcoords_location, 0.0f, 0.0f);
//	glVertex2f(-1.0f, -1.0f);
//
//	glVertexAttrib2f(texcoords_location, 1.0f, 0.0f);
//	glVertex2f(1.0f, -1.0f);
//
//	glVertexAttrib2f(texcoords_location, 1.0f, 1.0f);
//	glVertex2f(1.0f, 1.0f);
//
//	glVertexAttrib2f(texcoords_location, 0.0f, 1.0f);
//	glVertex2f(-1.0f, 1.0f);
//	glEnd();
}


