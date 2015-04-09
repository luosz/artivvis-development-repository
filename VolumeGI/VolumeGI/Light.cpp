#include "Light.h"

Light::Light(glm::vec3 pos, glm::vec3 dir)
{
	position = pos;
	direction = dir;

	width = 1024;
	height = 1024;

	lightCam.Init(width, height, 120.0f, position, direction);

	lightMap = new Texture2D(GL_RGBA32F, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	framebuffer = new Framebuffer(width, height, lightMap->ID);
}




//Texture2D* Light::GetLightMap(std::vector<Mesh> &meshes, float octreeSpan)
//{
//	framebuffer->Bind();
//
//	GLint oldViewPort[4];
//	glGetIntegerv(GL_VIEWPORT, oldViewPort);
//	glViewport(0, 0, width, height);
//
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//	DrawView(meshes, octreeSpan);
//
//	framebuffer->Unbind();	
//
//	glViewport(oldViewPort[0], oldViewPort[1], oldViewPort[2], oldViewPort[3]);	
//
//	return lightMap;
//}
//
//void Light::DrawView(std::vector<Mesh> &meshes, float octreeSpan)
//{
//	GLuint shaderProgramID = ShaderManager::UseShader(LightMapShader);
//
//	glm::mat4 modelMat = glm::mat4(1.0f);
//	GLuint uniformLoc = glGetUniformLocation(shaderProgramID, "model");
//	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &modelMat[0][0]);
//
//	uniformLoc = glGetUniformLocation (shaderProgramID, "proj");
//	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &lightCam.projMat[0][0]);
//
//	uniformLoc = glGetUniformLocation (shaderProgramID, "view");
//	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &lightCam.viewMat[0][0]);
//
//	uniformLoc = glGetUniformLocation (shaderProgramID, "span");
//	glUniform1f (uniformLoc, octreeSpan);
//
//	
//	for (int i=0; i<meshes.size(); i++)
//		meshes[i].Draw(shaderProgramID);	
//}