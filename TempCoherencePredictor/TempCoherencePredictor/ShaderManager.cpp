#include "ShaderManager.h"

// Loads and compiles all Shaders
void ShaderManager::Init()
{
	LoadShaders();
	CompileShaders();
	currentShader = TFShader;
}


// Reads in shaders, must load in the same order as listed in ShaderType enum
void ShaderManager::LoadShaders()
{
	Shader shader;

	shader.Load("SimpleVertShader.txt", "SimpleFragShader.txt");
	shaders.push_back(shader);

	shader.Load("TransFuncVertShader.txt", "TransFuncFragShader.txt");
	shaders.push_back(shader);

	shader.Load("BlockVertShader.txt", "BlockFragShader.txt");
	shaders.push_back(shader);

	shader.Load("GreyScaleVertShader.txt", "GreyScaleFragShader.txt");
	shaders.push_back(shader);

	shader.Load("VisibilityVertShader.txt", "VisibilityFragShader.txt");
	shaders.push_back(shader);
}


// Compiles each shader, put stops in shader.cpp code to debug errors
void ShaderManager::CompileShaders()
{
	for (unsigned int i=0; i<shaders.size(); i++)
	{
		shaders[i].CompileShaders();
	}
}


// Takes an enum of the type of shader desired, activates it and passes back an ID
GLuint ShaderManager::UseShader(ShaderType shaderType)
{
	GLuint ID = shaders[(int)shaderType].ID;
	glUseProgram (ID);

	return ID;
}