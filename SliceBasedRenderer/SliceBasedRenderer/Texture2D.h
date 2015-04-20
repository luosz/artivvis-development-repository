#ifndef TEXTURE_2D_H
#define TEXTURE_2D_H

#include <GL/glew.h>
#include <GL/freeglut.h>
#include "ShaderManager.h"

class Texture2D
{
public:
	GLuint ID;
	int xPixels, yPixels;

	Texture2D() {};
	Texture2D(GLint intFormat, GLsizei width, GLsizei height, GLenum format, GLenum type, GLvoid *data);
	void Render();
};

#endif