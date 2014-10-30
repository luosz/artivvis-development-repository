#include "OpenGLRenderer.h"

void OpenGLRenderer::Init(int screenWidth, int screenHeight, VolumeDataset &volume)
{
	volume.currTexture3D = volume.GenerateTexture();

	if (volume.timesteps > 1)
	{
		volume.voxelReader.CopyFileToBuffer(volume.memblock3D, 1);
		volume.nextTexture3D = volume.GenerateTexture();
	}

	raycaster = new GPURaycaster();
	raycaster->Init(screenWidth, screenHeight, volume);

	transferFunction.Init(" ");
}


void OpenGLRenderer::Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera)
{
	GLuint shaderProgramID = shaderManager.UseShader(shaderManager.currentShader);
	raycaster->Raycast(volume, transferFunction, shaderProgramID, camera);
	
}
