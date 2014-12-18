#include "BlockRaycaster.h"

BlockRaycaster::BlockRaycaster(int screenWidth, int screenHeight, VolumeDataset &volume)
{
	maxRaySteps = 90;
	rayStepSize = 0.005f;
	gradientStepSize = 0.005f;

	lightPosition = glm::vec3(-0.0f, -5.0f, 5.0f);

	int blockRes = 8;
	int xBlocks = glm::ceil((float)volume.xRes / (float)blockRes);
	int yBlocks = glm::ceil((float)volume.yRes / (float)blockRes);
	int zBlocks = glm::ceil((float)volume.zRes / (float)blockRes);
	numBlocks = xBlocks * yBlocks * zBlocks;

	float xVoxelWidth = 2.0f / (float) volume.xRes;
	float yVoxelWidth = 2.0f / (float) volume.yRes;
	float zVoxelWidth = 2.0f / (float) volume.zRes;

	blocks.reserve(numBlocks);

	for (int k=0; k<zBlocks; k++)
		for (int j=0; j<yBlocks; j++)
			for (int i=0; i<xBlocks; i++)
			{
				blocks.push_back(Block(blockRes, i, j, k, xVoxelWidth, yVoxelWidth, zVoxelWidth));
			}
}





// Messy but just inputs all data to shader
void BlockRaycaster::Raycast(VolumeDataset &volume, TransferFunction &transferFunction, GLuint shaderProgramID, Camera &camera)
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

	uniformLoc = glGetUniformLocation(shaderProgramID,"maxRaySteps");
	glUniform1i(uniformLoc, maxRaySteps);

	uniformLoc = glGetUniformLocation(shaderProgramID,"rayStepSize");
	glUniform1f(uniformLoc, rayStepSize);

	uniformLoc = glGetUniformLocation(shaderProgramID,"gradientStepSize");
	glUniform1f(uniformLoc, gradientStepSize);

	uniformLoc = glGetUniformLocation(shaderProgramID,"lightPosition");
	glUniform3f(uniformLoc, lightPosition.x, lightPosition.y, lightPosition.z);

	glActiveTexture (GL_TEXTURE2);
	uniformLoc = glGetUniformLocation(shaderProgramID,"transferFunc");
	glUniform1i(uniformLoc,2);
	glBindTexture (GL_TEXTURE_1D, transferFunction.tfTexture);

//	glEnable(GL_DEPTH_TEST);
//	glBlendFunc(GL_ONE, GL_ONE);
//	glEnable(GL_BLEND);

	// Final render is the front faces of a cube rendered
	

	for (int i=0; i<numBlocks; i++)
	{
		glBegin(GL_POLYGON);
		glVertex3f(blocks[i].vertices[0].x, blocks[i].vertices[0].y, blocks[i].vertices[0].z);
		glVertex3f(blocks[i].vertices[1].x, blocks[i].vertices[1].y, blocks[i].vertices[1].z);
		glVertex3f(blocks[i].vertices[3].x, blocks[i].vertices[3].y, blocks[i].vertices[3].z);
		glVertex3f(blocks[i].vertices[2].x, blocks[i].vertices[2].y, blocks[i].vertices[2].z);
	
		glVertex3f(blocks[i].vertices[4].x, blocks[i].vertices[4].y, blocks[i].vertices[4].z);
		glVertex3f(blocks[i].vertices[5].x, blocks[i].vertices[5].y, blocks[i].vertices[5].z);
		glVertex3f(blocks[i].vertices[7].x, blocks[i].vertices[7].y, blocks[i].vertices[7].z);
		glVertex3f(blocks[i].vertices[6].x, blocks[i].vertices[6].y, blocks[i].vertices[6].z);
	
	
		glVertex3f(blocks[i].vertices[0].x, blocks[i].vertices[0].y, blocks[i].vertices[0].z);
		glVertex3f(blocks[i].vertices[1].x, blocks[i].vertices[1].y, blocks[i].vertices[1].z);
		glVertex3f(blocks[i].vertices[5].x, blocks[i].vertices[5].y, blocks[i].vertices[5].z);
		glVertex3f(blocks[i].vertices[4].x, blocks[i].vertices[4].y, blocks[i].vertices[4].z);
	
		glVertex3f(blocks[i].vertices[2].x, blocks[i].vertices[2].y, blocks[i].vertices[2].z);
		glVertex3f(blocks[i].vertices[3].x, blocks[i].vertices[3].y, blocks[i].vertices[3].z);
		glVertex3f(blocks[i].vertices[7].x, blocks[i].vertices[7].y, blocks[i].vertices[7].z);
		glVertex3f(blocks[i].vertices[6].x, blocks[i].vertices[6].y, blocks[i].vertices[6].z);
	
	
		glVertex3f(blocks[i].vertices[0].x, blocks[i].vertices[0].y, blocks[i].vertices[0].z);
		glVertex3f(blocks[i].vertices[2].x, blocks[i].vertices[2].y, blocks[i].vertices[2].z);
		glVertex3f(blocks[i].vertices[6].x, blocks[i].vertices[6].y, blocks[i].vertices[6].z);
		glVertex3f(blocks[i].vertices[4].x, blocks[i].vertices[4].y, blocks[i].vertices[4].z);

		glVertex3f(blocks[i].vertices[1].x, blocks[i].vertices[1].y, blocks[i].vertices[1].z);
		glVertex3f(blocks[i].vertices[3].x, blocks[i].vertices[3].y, blocks[i].vertices[3].z);
		glVertex3f(blocks[i].vertices[7].x, blocks[i].vertices[7].y, blocks[i].vertices[7].z);
		glVertex3f(blocks[i].vertices[5].x, blocks[i].vertices[5].y, blocks[i].vertices[5].z);
		glEnd();
	}

	

	glBindTexture(GL_TEXTURE_3D, 0);
}
