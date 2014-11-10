#include "VisibilityRenderer.h"

void VisibilityRenderer::Init(int screenWidth, int screenHeight, VolumeDataset &volume)
{
	volume.currTexture3D = volume.GenerateTexture();

	if (volume.timesteps > 1)
	{
		volume.voxelReader.CopyFileToBuffer(volume.memblock3D, 1);
		volume.nextTexture3D = volume.GenerateTexture();
	}

	raycaster = new VisibilityRaycaster();
	raycaster->Init(screenWidth, screenHeight, volume);

	transferFunction.Init(" ", volume);
}


void VisibilityRenderer::Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera)
{
	GLuint shaderProgramID = shaderManager.UseShader(VisibilityHistShader);
	raycaster->Raycast(volume, transferFunction, shaderProgramID, camera);
	
}
