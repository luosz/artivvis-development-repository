#ifndef SHADER_MANAGER_H
#define SHADER_MANAGER_H

#include "Shader.h"
#include <vector>
#include "GLM.h"

enum ShaderType {SimpleShader, TextureShader, VolumeShader, RaycastShader, LightingShader};

class ShaderManager
{
public:
	void Init();
	GLuint UseShader(ShaderType shaderType);

private:
	std::vector<Shader> shaders;
	std::vector<GLuint> shaderProgramIDs;

	void LoadShaders();
	void CompileShaders();
};

#endif