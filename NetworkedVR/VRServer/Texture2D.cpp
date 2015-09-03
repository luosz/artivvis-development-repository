#include "Texture2D.h"

Texture2D::Texture2D(GLint intFormat, GLsizei width, GLsizei height, GLenum format, GLenum type, GLvoid *data)
{
	xPixels = width;
	yPixels = height;

	glGenTextures(1, &ID);
	glBindTexture(GL_TEXTURE_2D, ID);
	glTexImage2D(GL_TEXTURE_2D, 0, intFormat, width, height, 0, format, type, data);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glBindTexture(GL_TEXTURE_2D, 0);
}



void Texture2D::Render()
{
	GLuint shaderProgramID = ShaderManager::UseShader(TextureShader);

	glActiveTexture (GL_TEXTURE0);
	int texLoc = glGetUniformLocation(shaderProgramID, "texColor");
	glUniform1i(texLoc,0);
	glBindTexture (GL_TEXTURE_2D, ID);
	int texcoords_location = glGetAttribLocation (shaderProgramID, "vTexture");

	glBegin(GL_QUADS);
	glVertexAttrib2f(texcoords_location, 0.0f, 0.0f);
	glVertex2f(-1.0f, -1.0f);

	glVertexAttrib2f(texcoords_location, 1.0f, 0.0f);
	glVertex2f(1.0f, -1.0f);

	glVertexAttrib2f(texcoords_location, 1.0f, 1.0f);
	glVertex2f(1.0f, 1.0f);

	glVertexAttrib2f(texcoords_location, 0.0f, 1.0f);
	glVertex2f(-1.0f, 1.0f);
	glEnd();
}


Texture2D::~Texture2D()
{
//	glDeleteTextures(1, &ID);
}