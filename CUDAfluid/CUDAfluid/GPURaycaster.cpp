#include "GPURaycaster.h"

void GPURaycaster::Init(int screenWidth, int screenHeight)
{
	maxRaySteps = 1000;
	rayStepSize = 0.005f;
	gradientStepSize = 0.005f;

	GenerateTexture();
}

void GPURaycaster::GenerateTexture()
{
	glEnable(GL_TEXTURE_3D);
	glGenTextures(1, &texID);
	glBindTexture(GL_TEXTURE_3D, texID);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

//	glPixelStorei(GL_UNPACK_ALIGNMENT, 2);

	glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, gridXRes, gridYRes, gridZRes, 0,  GL_RED, GL_FLOAT, NULL);	

	glBindTexture(GL_TEXTURE_3D, 0);

	glEnable(GL_TEXTURE_3D);
	glGenTextures(1, &tex2ID);
	glBindTexture(GL_TEXTURE_3D, tex2ID);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, gridXRes, gridYRes, gridZRes, 0,  GL_RED, GL_FLOAT, NULL);	

	glBindTexture(GL_TEXTURE_3D, 0);
}





// Messy but just inputs all data to shader
void GPURaycaster::Raycast(GLuint shaderProgramID, Camera &camera, std::vector<float> &vector1, std::vector<float> &vector2)
{
	glBindTexture (GL_TEXTURE_3D, texID);
	glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, gridXRes, gridYRes, gridZRes, GL_RED, GL_FLOAT, &vector1[0]);
	glBindTexture(GL_TEXTURE_3D, 0);

	glBindTexture (GL_TEXTURE_3D, tex2ID);
	glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, gridXRes, gridYRes, gridZRes, GL_RED, GL_FLOAT, &vector2[0]);
	glBindTexture(GL_TEXTURE_3D, 0);

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
	glBindTexture (GL_TEXTURE_3D, texID);

	uniformLoc = glGetUniformLocation(shaderProgramID,"camPos");
	glUniform3f(uniformLoc, camera.position.x, camera.position.y, camera.position.z);

	uniformLoc = glGetUniformLocation(shaderProgramID,"maxRaySteps");
	glUniform1i(uniformLoc, maxRaySteps);

	uniformLoc = glGetUniformLocation(shaderProgramID,"rayStepSize");
	glUniform1f(uniformLoc, rayStepSize);

	uniformLoc = glGetUniformLocation(shaderProgramID,"gradientStepSize");
	glUniform1f(uniformLoc, gradientStepSize);

//	glBind Texture(GL_TEXTURE_3D, 0);
	glActiveTexture(GL_TEXTURE1);
	uniformLoc = glGetUniformLocation(shaderProgramID,"temperature");
	glUniform1i(uniformLoc,1);
	glBindTexture (GL_TEXTURE_3D, tex2ID);

	// Final render is the front faces of a cube rendered
	glBegin(GL_QUADS);

	// Front Face
	glVertex3f(-1.0f, -1.0f,  1.0f);
	glVertex3f( 1.0f, -1.0f,  1.0f);
	glVertex3f( 1.0f,  1.0f,  1.0f);
	glVertex3f(-1.0f,  1.0f,  1.0f);
 
	// Back Face
	glVertex3f(-1.0f, -1.0f, -1.0f);
	glVertex3f(-1.0f,  1.0f, -1.0f);
	glVertex3f( 1.0f,  1.0f, -1.0f);
	glVertex3f( 1.0f, -1.0f, -1.0f);
 
	// Top Face
	glVertex3f(-1.0f,  1.0f, -1.0f);
	glVertex3f(-1.0f,  1.0f,  1.0f);
	glVertex3f( 1.0f,  1.0f,  1.0f);
	glVertex3f( 1.0f,  1.0f, -1.0f);
	
	// Bottom Face
	glVertex3f(-1.0f, -1.0f, -1.0f);
	glVertex3f( 1.0f, -1.0f, -1.0f);
	glVertex3f( 1.0f, -1.0f,  1.0f);
	glVertex3f(-1.0f, -1.0f,  1.0f);
 
	// Right face
	glVertex3f( 1.0f, -1.0f, -1.0f);
	glVertex3f( 1.0f,  1.0f, -1.0f);
	glVertex3f( 1.0f,  1.0f,  1.0f);
	glVertex3f( 1.0f, -1.0f,  1.0f);
 
	// Left Face
	glVertex3f(-1.0f, -1.0f, -1.0f);
	glVertex3f(-1.0f, -1.0f,  1.0f);
	glVertex3f(-1.0f,  1.0f,  1.0f);
	glVertex3f(-1.0f,  1.0f, -1.0f);

	glEnd();

	glBindTexture(GL_TEXTURE_3D, 0);
	
}
