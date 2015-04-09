#ifndef RENDERER_H
#define RENDERER_H

#include "Camera.h"
#include "ShaderManager.h"
#include "Octree.h"
#include "Light.h"
#include "Texture2D.h"
#include "VBO.h"
#include "Raycaster.h"
#include "VolumeDataset.h"

class Renderer
{
public:
	int screenWidth, screenHeight;

	Camera camera;
	Octree octree;
	Raycaster *raycaster;
	Light *light;
	VolumeDataset volume;
	TransferFunction transferFunction;

	Framebuffer *defFramebuffer;
	Texture2D *defPosTex, *defNormTex, *defColorTex, *defSpecTex;

	VBO screenVBO;

	void Init(int screenWidth_, int screenHeight_);
	void Draw();

	void InitScreenSpace();

	void RenderSimple();
	void RenderGI();

	void DeferredPass();
	void FinalGIPass();
};

#endif