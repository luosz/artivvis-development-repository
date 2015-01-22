#include "TomsOGLRenderer.h"

TomsOGLRenderer::TomsOGLRenderer(int screenWidth, int screenHeight, VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera)   :   OpenGLRenderer(screenWidth, screenHeight, volume, shaderManager, camera)
{

}


void TomsOGLRenderer::Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera)
{
	//visibilityHistogram.CalculateHistogram(volume, transferFunction.tfTexture, shaderManager, camera);
	//optimizer->Optimize();
	OpenGLRenderer::Draw(volume, shaderManager, camera);
}
