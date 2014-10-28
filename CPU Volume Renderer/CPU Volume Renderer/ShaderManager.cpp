#include "ShaderManager.h"

void ShaderManager::Init()
{
	LoadShaders();
	CompileShaders();
}


// Reads in shaders, must load in the same order as listed in ShaderType enum
void ShaderManager::LoadShaders()
{
	Shader shader;

	shader.Load("SimpleVertShader.txt", "SimpleFragShader.txt");
	shaders.push_back(shader);

	shader.Load("TextureVertShader.txt", "TextureFragShader.txt");
	shaders.push_back(shader);

	shader.Load("VolumeVertShader.txt", "VolumeFragShader.txt");
	shaders.push_back(shader);

	shader.Load("RaycastVertShader.txt", "RaycastFragShader.txt");
	shaders.push_back(shader);

	shader.Load("LightingVertShader.txt", "LightingFragShader.txt");
	shaders.push_back(shader);

	CompileShaders();
}



void ShaderManager::CompileShaders()
{
	shaderProgramIDs.resize(shaders.size());
	for (unsigned int i=0; i<shaders.size(); i++)
	{
		shaderProgramIDs[i] = shaders[i].CompileShaders();
	}
}



GLuint ShaderManager::UseShader(ShaderType shaderType)
{
	GLuint ID = shaderProgramIDs[(int)shaderType];
	glUseProgram (ID);

	return ID;
}