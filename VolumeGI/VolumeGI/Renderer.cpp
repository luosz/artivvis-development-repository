#include "Renderer.h"

void Renderer::Init(int screenWidth_, int screenHeight_)
{
	screenWidth = screenWidth_;
	screenHeight = screenHeight_;

	ShaderManager::Init();
	camera.Init(screenWidth, screenHeight);
	volume.Init();

	raycaster = new Raycaster(screenWidth, screenHeight, volume);
	transferFunction.Init(" ", volume);
	
	InitScreenSpace();

	light = new Light(glm::vec3(7.5f, 7.5f, 0.0f), glm::vec3(-1.0f, -0.5f, 0.0f));


	
}

void Renderer::InitScreenSpace()
{
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec2> texCoords;

	texCoords.push_back(glm::vec2(0.0f, 0.0f));
	vertices.push_back(glm::vec3(-1.0f, -1.0f, 0.0f));

	texCoords.push_back(glm::vec2(1.0f, 0.0f));
	vertices.push_back(glm::vec3(1.0f, -1.0f, 0.0f));

	texCoords.push_back(glm::vec2(1.0f,1.0f));
	vertices.push_back(glm::vec3(1.0f, 1.0f, 0.0f));


	texCoords.push_back(glm::vec2(1.0f,1.0f));
	vertices.push_back(glm::vec3(1.0f, 1.0f, 0.0f));

	texCoords.push_back(glm::vec2(0.0f, 1.0f));
	vertices.push_back(glm::vec3(-1.0f, 1.0f, 0.0f));

	texCoords.push_back(glm::vec2(0.0f, 0.0f));
	vertices.push_back(glm::vec3(-1.0f, -1.0f, 0.0f));

	screenVBO.Load(vertices, vertices, texCoords);


	defPosTex = new Texture2D(GL_RGBA32F, screenWidth, screenHeight, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	defNormTex = new Texture2D(GL_RGBA32F, screenWidth, screenHeight, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	defColorTex = new Texture2D(GL_RGBA32F, screenWidth, screenHeight, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	defSpecTex = new Texture2D(GL_RGBA32F, screenWidth, screenHeight, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	defFramebuffer = new Framebuffer(screenWidth, screenHeight, defPosTex->ID);
}


void Renderer::Draw()
{
	camera.Update();
	volume.Update();

	RenderGI();

}



void Renderer::RenderGI()
{

	RenderSimple();
}

void Renderer::FinalGIPass()
{

}


void Renderer::DeferredPass()
{
	
}

void Renderer::RenderSimple()
{
	GLuint shaderProgramID = ShaderManager::UseShader(TFShader);
	raycaster->Raycast(transferFunction, shaderProgramID, camera, volume.currTexture3D);
}
