#ifndef LIGHT_H
#define LIGHT_H

#include "GLM.h"
#include "Camera.h"
#include "Framebuffer.h"
#include "Texture2D.h"

class Light
{
public:
	glm::vec3 position;
	glm::vec3 direction;
	int width, height;

	Camera lightCam;
	Framebuffer *framebuffer;
	Texture2D *lightMap;

	Light(glm::vec3 pos, glm::vec3 dir);

//	Texture2D* GetLightMap(std::vector<Mesh> &meshes, float octreeSpan);
//	void DrawView(std::vector<Mesh> &meshes, float octreeSpan);
};

#endif