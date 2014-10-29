#ifndef SHADER_MANAGER_H
#define SHADER_MANAGER_H

#include "Shader.h"
#include <vector>
#include "GLM.h"

enum ShaderType {	SimpleShader, TextureShader, VolumeShader, RaycastShader, LightingShader, OpacityShader, 
					DepthShader, NormalsShader, DiffuseShader, ContourShader, ComparisonShader, LineDepthShader, 
					XToonShader, MarchingSurfaceShader, ShadowShader, SmokeShader, TFShader};

class ShaderManager
{
public:
	ShaderType currentShader;

	void Init();
	GLuint UseShader(ShaderType shaderType);

private:
	std::vector<Shader> shaders;

	void LoadShaders();
	void CompileShaders();
};

#endif