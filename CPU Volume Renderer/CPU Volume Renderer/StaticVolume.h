#ifndef STATICVOLUME_H
#define STATICVOLUME_H

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>
#include <fstream>
#include <string>
#include "GLM.h"
#include "Camera.h"
#include <vector>

class StaticVolume
{
// Defines specific to input file

#define NUM_SLICES 113
#define X_PIXELS 256
#define Y_PIXELS 256
#define FILE_PATH "../../Samples/CThead/CTheadRaw/CThead."

public:
	char *memblock3D;
	char *ordered;
	GLuint texture3D;
	int xRes, yRes, zRes;
	std::vector<short> shorts;

	void ReadFiles();
};


#endif