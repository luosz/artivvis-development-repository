#ifndef VAO_H
#define VAO_H

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <vector>
#include "GLM.h"

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

class VBO
{
public:
	void Load(std::vector<glm::vec3> &positions, std::vector<glm::vec3> &normals, std::vector<glm::vec2> &texCoords);
	void Draw(GLuint shaderProgramID);

private:
	GLuint ID;
	int count;
};

#endif