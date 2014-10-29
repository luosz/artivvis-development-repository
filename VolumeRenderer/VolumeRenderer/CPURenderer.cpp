#include "CPURenderer.h"

void CPURenderer::Init(int screenWidth, int screenHeight, VolumeDataset &volume)
{
	raycaster = new CPURaycaster();
	raycaster->Init(screenWidth, screenHeight, volume);
}


void CPURenderer::Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera)
{
	GLuint shaderProgramID = shaderManager.UseShader(TextureShader);
	raycaster->Raycast(volume, transferFunction, shaderProgramID, camera);

	
}