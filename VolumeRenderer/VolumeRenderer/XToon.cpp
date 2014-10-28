#include "XToon.h"

void XToon::ReadTexture()
{
	std::string fileName = "metallic.png";

	glGenTextures(1, &toonTexture);
	glBindTexture (GL_TEXTURE_2D, toonTexture);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	ILuint texID;
	ILboolean success;

	ilInit();
	ilGenImages(1,&texID);
	ilBindImage(texID);

	success = ilLoadImage((const ILstring)fileName.c_str());

	if(success)
	{
		success = ilConvertImage(IL_RGBA,IL_UNSIGNED_BYTE);
		glTexImage2D(GL_TEXTURE_2D, 0, ilGetInteger(IL_IMAGE_BPP), ilGetInteger(IL_IMAGE_WIDTH), ilGetInteger(IL_IMAGE_HEIGHT),0, ilGetInteger(IL_IMAGE_FORMAT), GL_UNSIGNED_BYTE, ilGetData());
		textureWidth = ilGetInteger(IL_IMAGE_WIDTH);
		textureHeight = ilGetInteger(IL_IMAGE_HEIGHT);
	}

	ilDeleteImages(1,&texID);
}



void XToon::Render(ShaderManager &shaderManager)
{
	GLuint shaderProgramID = shaderManager.UseShader(TextureShader);

	glActiveTexture (GL_TEXTURE0);
	int uniformLocation = glGetUniformLocation(shaderProgramID,"texColor");
	glUniform1i(uniformLocation, 0);
	glBindTexture (GL_TEXTURE_2D, toonTexture);

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