#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <GL/glew.h>
#include <GL/freeglut.h>

class Framebuffer
{
public:
	GLuint ID;
	int xPixels, yPixels;

	Framebuffer(int xPixels_, int yPixels_, GLuint tex)
	{
		xPixels = xPixels_;
		yPixels = yPixels_;

		// Generate framebuffer
		glGenFramebuffers (1, &ID);
		glBindFramebuffer (GL_FRAMEBUFFER, ID);

		GLuint rb = 0;
		glGenRenderbuffers (1, &rb);
		glBindRenderbuffer (GL_RENDERBUFFER, rb);
		glRenderbufferStorage (GL_RENDERBUFFER, GL_DEPTH_COMPONENT, xPixels, yPixels);
		glFramebufferRenderbuffer (GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb);
		glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);

		glBindFramebuffer (GL_FRAMEBUFFER, 0);
	}

	void Bind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, ID);
	}

	void Unbind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
};

#endif