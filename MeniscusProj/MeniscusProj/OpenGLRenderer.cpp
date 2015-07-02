#include "OpenGLRenderer.h"

OpenGLRenderer::OpenGLRenderer(int screenWidth, int screenHeight, VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera)
{
	currTexture3D = GenerateTexture(volume);

	raycaster = new GPURaycaster(screenWidth, screenHeight, volume);
//	transferFunction.Init(" ", volume);

	tfBandWidth = 0.1f; 
	tfBandPos = 0.5f;
}


// Updates the texture by copying the block corresponding to the current timestep to GPU memory
void OpenGLRenderer::UpdateTexture(int currentTimestep, VolumeDataset &volume)
{
	glBindTexture(GL_TEXTURE_3D, currTexture3D);

	if (!volume.littleEndian)
		glPixelStoref(GL_UNPACK_SWAP_BYTES, true);

	if (volume.elementType == "MET_UCHAR")
		glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, volume.xRes, volume.yRes, volume.zRes, 0,  GL_RED, GL_UNSIGNED_BYTE, volume.memblock3D + (textureSize * currentTimestep));

	else if (volume.elementType == "SHORT")
		glTexImage3D(GL_TEXTURE_3D, 0, GL_R16F, volume.xRes, volume.yRes, volume.zRes, 0, GL_RED, GL_UNSIGNED_SHORT, volume.memblock3D + (textureSize * currentTimestep));

	else if (volume.elementType == "FLOAT")
		glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, volume.xRes, volume.yRes, volume.zRes, 0, GL_RED, GL_FLOAT, volume.memblock3D + (textureSize * currentTimestep));

	glPixelStoref(GL_UNPACK_SWAP_BYTES, false);

	glBindTexture(GL_TEXTURE_3D, 0);
}


// Generates the original 3D texture
GLuint OpenGLRenderer::GenerateTexture(VolumeDataset &volume)
{
	GLuint tex;
	textureSize = volume.xRes * volume.yRes * volume.zRes * volume.bytesPerElement;

	glEnable(GL_TEXTURE_3D);
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_3D, tex);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);


	// Reverses endianness in copy
	if (!volume.littleEndian)
		glPixelStoref(GL_UNPACK_SWAP_BYTES, true);

	if (volume.elementType == "MET_UCHAR")
		glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, volume.xRes, volume.yRes, volume.zRes, 0,  GL_RED, GL_UNSIGNED_BYTE, volume.memblock3D);

	else if (volume.elementType == "SHORT")
		glTexImage3D(GL_TEXTURE_3D, 0, GL_R16F, volume.xRes, volume.yRes, volume.zRes, 0, GL_RED, GL_UNSIGNED_SHORT, volume.memblock3D);

	else if (volume.elementType == "FLOAT")
		glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, volume.xRes, volume.yRes, volume.zRes, 0, GL_RED, GL_FLOAT, volume.memblock3D);

	glPixelStoref(GL_UNPACK_SWAP_BYTES, false);
	
	glBindTexture(GL_TEXTURE_3D, 0);

	return tex;
}


void OpenGLRenderer::Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera, ClipPlane &clipPlane, bool focused, bool removed, float sphereRadius, glm::vec3 &spherePoint)
{
	GLuint shaderProgramID = shaderManager.UseShader(TFShader);
		
	GLint uniformLoc = glGetUniformLocation(shaderProgramID, "clipPlanePos");
	glUniform3f(uniformLoc, clipPlane.point.x, clipPlane.point.y, clipPlane.point.z);

//	uniformLoc = glGetUniformLocation(shaderProgramID,"clipPlaneNormal");
//	glUniform3f(uniformLoc, clipPlane.normal.x, clipPlane.normal.y, clipPlane.normal.z);

	uniformLoc = glGetUniformLocation(shaderProgramID, "focused");
	glUniform1i(uniformLoc, focused);

	uniformLoc = glGetUniformLocation(shaderProgramID, "removed");
	glUniform1i(uniformLoc, removed);

	uniformLoc = glGetUniformLocation(shaderProgramID, "sphereRadius");
	glUniform1f(uniformLoc, sphereRadius);

	uniformLoc = glGetUniformLocation(shaderProgramID,"spherePoint");
	glUniform3f(uniformLoc, spherePoint.x, spherePoint.y, spherePoint.z);

	uniformLoc = glGetUniformLocation(shaderProgramID, "clipAxis");
	glUniform1i(uniformLoc, 2);
	if (clipPlane.normal.x == 1.0f)
		glUniform1i(uniformLoc, 0);
	else if (clipPlane.normal.y == 1.0f)
		glUniform1i(uniformLoc, 1);
	else if (clipPlane.normal.z == 1.0f)
		glUniform1i(uniformLoc, 2);

	uniformLoc = glGetUniformLocation(shaderProgramID, "tfBandWidth");
	glUniform1f(uniformLoc, tfBandWidth);

	uniformLoc = glGetUniformLocation(shaderProgramID, "tfBandPos");
	glUniform1f(uniformLoc, tfBandPos);


	raycaster->Raycast(currTexture3D, transferFunction, shaderProgramID, camera);
}