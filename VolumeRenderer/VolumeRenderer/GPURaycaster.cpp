#include "GPURaycaster.h"

GPURaycaster::GPURaycaster(int screenWidth, int screenHeight, VolumeDataset &volume)
{
	maxRaySteps = 1000;
	rayStepSize = 0.005f;
	gradientStepSize = 0.005f;
	contourThreshold = 0.0f;
	lightPosition = glm::vec3(-0.0f, -5.0f, 5.0f);

	minRange = 0.0f;
	cutOff = 0.5f;
	maxRange = 1.0f;
	
	int numOpacityDivisions = 4;

	for (int i=0; i<numOpacityDivisions; i++)
	{
		opacityDivisions.push_back(glm::vec2(0.0f, 0.0f));
		opacities.push_back(0.0f);
	}

	opacityDivisions[0] = glm::vec2(0.14f, 0.41f);
	opacityDivisions[1] = glm::vec2(0.41f, 0.5f);
	opacityDivisions[2] = glm::vec2(0.78f, 0.99f);
	opacityDivisions[3] = glm::vec2(1.0f, 1.0f);

	opacities[0] = 0.4f;
	opacities[1] = 0.6f;

	clipPlaneDistance = 0.0f;
	clipPlaneNormal = glm::vec3(0.0f, 0.0f, 1.0f);

	xToon.ReadTexture();
//	GenerateTextures(volume);
}

//void GPURaycaster::GenerateTextures(VolumeDataset &volume)
//{
//	textures.resize(volume.timesteps);
//
//	int textureSize = volume.xRes * volume.yRes * volume.zRes * volume.bytesPerElement;
//
//	for (int i=0; i<volume.timesteps; i++)
//	{
//		glEnable(GL_TEXTURE_3D);
//		glGenTextures(1, &textures[i]);
//		glBindTexture(GL_TEXTURE_3D, textures[i]);
//		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
//		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
//		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
//
//		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
//
//
//		// Reverses endianness in copy
//		if (!volume.littleEndian)
//			glPixelStoref(GL_UNPACK_SWAP_BYTES, true);
//
//		if (volume.elementType == "MET_UCHAR")
//			glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, volume.xRes, volume.yRes, volume.zRes, 0,  GL_RED, GL_UNSIGNED_BYTE, volume.memblock3D + (i * textureSize));
//
//		else if (volume.elementType == "SHORT")
//			glTexImage3D(GL_TEXTURE_3D, 0, GL_R16F, volume.xRes, volume.yRes, volume.zRes, 0, GL_RED, GL_UNSIGNED_SHORT, volume.memblock3D + (i * textureSize));
//
//		else if (volume.elementType == "FLOAT")
//			glTexImage3D(GL_TEXTURE_3D, 0, GL_R16F, volume.xRes, volume.yRes, volume.zRes, 0, GL_RED, GL_FLOAT, volume.memblock3D + (i * textureSize));
//
//		glPixelStoref(GL_UNPACK_SWAP_BYTES, false);
//		
//
//		glBindTexture(GL_TEXTURE_3D, 0);
//	}
//}





// Messy but just inputs all data to shader
void GPURaycaster::Raycast(VolumeDataset &volume, TransferFunction &transferFunction, GLuint shaderProgramID, Camera &camera)
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

	uniformLoc = glGetUniformLocation(shaderProgramID,"contourThreshold");
	glUniform1f(uniformLoc, contourThreshold);

	glActiveTexture (GL_TEXTURE1);
	uniformLoc = glGetUniformLocation(shaderProgramID,"xToonTexture");
	glUniform1i(uniformLoc,1);
	glBindTexture (GL_TEXTURE_2D, xToon.toonTexture);

	uniformLoc = glGetUniformLocation(shaderProgramID, "toonTextureWidth");
	glUniform1i(uniformLoc, xToon.textureWidth);

	uniformLoc = glGetUniformLocation(shaderProgramID, "toonTextureHeight");
	glUniform1i(uniformLoc, xToon.textureHeight);


	uniformLoc = glGetUniformLocation(shaderProgramID,"lightPosition");
	glUniform3f(uniformLoc, lightPosition.x, lightPosition.y, lightPosition.z);

	glActiveTexture (GL_TEXTURE2);
	uniformLoc = glGetUniformLocation(shaderProgramID,"transferFunc");
	glUniform1i(uniformLoc,2);
	glBindTexture (GL_TEXTURE_1D, transferFunction.tfTexture);


	for (int i=0; i<opacityDivisions.size(); i++)
	{
		uniformLoc = glGetUniformLocation(shaderProgramID, (std::string("division" + std::to_string(i+1))).c_str());
		glUniform2f(uniformLoc, opacityDivisions[i].x, opacityDivisions[i].y);

		uniformLoc = glGetUniformLocation(shaderProgramID, (std::string("opacity" + std::to_string(i+1))).c_str());
		glUniform1f(uniformLoc, opacities[i]);
	}

	clipPlaneDistance = glm::clamp(clipPlaneDistance, 0.0f, 2.0f);
	uniformLoc = glGetUniformLocation(shaderProgramID, "clipPlaneDistance");
	glUniform1f(uniformLoc, clipPlaneDistance);

	uniformLoc = glGetUniformLocation(shaderProgramID,"clipPlaneNormal");
	glUniform3f(uniformLoc, clipPlaneNormal.x, clipPlaneNormal.y, clipPlaneNormal.z);


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
