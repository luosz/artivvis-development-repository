#include "Raycaster.h"

Raycaster::Raycaster(int screenWidth, int screenHeight, VolumeDataset &volume)
{
	maxRaySteps = 1000;
	rayStepSize = 0.005f;
	gradientStepSize = 0.005f;
	lightPosition = glm::vec3(-0.0f, -5.0f, 5.0f);

	minRange = 0.0f;
	cutOff = 0.5f;
	maxRange = 1.0f;
}





// Messy but just inputs all data to shader
void Raycaster::Raycast(VolumeDataset &volume, TransferFunction &transferFunction, GLuint shaderProgramID, Camera &camera)
{
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

	uniformLoc = glGetUniformLocation(shaderProgramID,"camPos");
	glUniform3f(uniformLoc, camera.position.x, camera.position.y, camera.position.z);

	uniformLoc = glGetUniformLocation(shaderProgramID,"minRange");
	glUniform1f(uniformLoc, minRange);

	uniformLoc = glGetUniformLocation(shaderProgramID,"cutOff");
	glUniform1f(uniformLoc, cutOff);

	uniformLoc = glGetUniformLocation(shaderProgramID,"maxRange");
	glUniform1f(uniformLoc, maxRange);

	uniformLoc = glGetUniformLocation(shaderProgramID,"maxRaySteps");
	glUniform1i(uniformLoc, maxRaySteps);

	uniformLoc = glGetUniformLocation(shaderProgramID,"rayStepSize");
	glUniform1f(uniformLoc, rayStepSize);

	uniformLoc = glGetUniformLocation(shaderProgramID,"gradientStepSize");
	glUniform1f(uniformLoc, gradientStepSize);


	uniformLoc = glGetUniformLocation(shaderProgramID,"lightPosition");
	glUniform3f(uniformLoc, lightPosition.x, lightPosition.y, lightPosition.z);

	glActiveTexture (GL_TEXTURE1);
	uniformLoc = glGetUniformLocation(shaderProgramID,"transferFunc");
	glUniform1i(uniformLoc,1);
	glBindTexture (GL_TEXTURE_1D, transferFunction.tfTexture);


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
