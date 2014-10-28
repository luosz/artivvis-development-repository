#ifndef XTOON_H
#define XTOON_H

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "GLM.h"
#include "Camera.h"
#include "ShaderManager.h"
#include <vector>
#include <il.h>
#include <ilu.h>
#include <ilut.h>

class XToon
{
public:
	GLubyte *toonBuffer;
	GLuint toonTexture;

	int textureWidth, textureHeight;

	void ReadTexture();
	void Render(ShaderManager &shaderManager);

};



#endif 
