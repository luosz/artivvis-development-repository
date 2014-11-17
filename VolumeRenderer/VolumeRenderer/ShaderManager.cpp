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

	shader.Load("TextureVertShader.txt", "TextureFragShader.txt");
	shaders.push_back(shader);

	shader.Load("VolumeVertShader.txt", "VolumeFragShader.txt");
	shaders.push_back(shader);

	shader.Load("RaycastVertShader.txt", "RaycastFragShader.txt");
	shaders.push_back(shader);

	shader.Load("LightingVertShader.txt", "LightingFragShader.txt");
	shaders.push_back(shader);

	shader.Load("OpacityVertShader.txt", "OpacityFragShader.txt");
	shaders.push_back(shader);

	shader.Load("DepthVertShader.txt", "DepthFragShader.txt");
	shaders.push_back(shader);

	shader.Load("NormalsVertShader.txt", "NormalsFragShader.txt");
	shaders.push_back(shader);

	shader.Load("DiffuseVertShader.txt", "DiffuseFragShader.txt");
	shaders.push_back(shader);

	shader.Load("ContourVertShader.txt", "ContourFragShader.txt");
	shaders.push_back(shader);

	shader.Load("ComparisonVertShader.txt", "ComparisonFragShader.txt");
	shaders.push_back(shader);

	shader.Load("LineDepthVertShader.txt", "LineDepthFragShader.txt");
	shaders.push_back(shader);

	shader.Load("XToonVertShader.txt", "XToonFragShader.txt");
	shaders.push_back(shader);

	shader.Load("MarchingSurfaceVertShader.txt", "MarchingSurfaceFragShader.txt");
	shaders.push_back(shader);

	shader.Load("ShadowVertShader.txt", "ShadowFragShader.txt");
	shaders.push_back(shader);

	shader.Load("SmokeVertShader.txt", "SmokeFragShader.txt");
	shaders.push_back(shader);

	shader.Load("TransFuncVertShader.txt", "TransFuncFragShader.txt");
	shaders.push_back(shader);

	shader.Load("VisibilityVertShader.txt", "VisibilityFragShader.txt");
	shaders.push_back(shader);

	shader.Load("RegionVisibilityVertShader.txt", "RegionVisibilityFragShader.txt");
	shaders.push_back(shader);

	CompileShaders();
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