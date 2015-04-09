#include "VBO.h"

void VBO::Load(std::vector<glm::vec3> &positions, std::vector<glm::vec3> &normals, std::vector<glm::vec2> &texCoords)
{
	count = positions.size();

	//Calc Array Sizes
	GLuint vertexArraySize = count * sizeof(glm::vec3);
	GLuint normalArraySize = count * sizeof(glm::vec3);
	GLuint texCoordArraySize = count * sizeof(glm::vec2);

	//Initialize VBO
	glGenBuffers( 1, &ID );
	glBindBuffer( GL_ARRAY_BUFFER, ID );
	glBufferData( GL_ARRAY_BUFFER, vertexArraySize + normalArraySize + texCoordArraySize, NULL, GL_STATIC_DRAW );
	glBufferSubData( GL_ARRAY_BUFFER, 0, vertexArraySize, (const GLvoid*)(&positions[0]) );
	glBufferSubData( GL_ARRAY_BUFFER, vertexArraySize, normalArraySize, (const GLvoid*)(&normals[0]));
	glBufferSubData( GL_ARRAY_BUFFER, vertexArraySize + normalArraySize, texCoordArraySize, (const GLvoid*)(&texCoords[0]));

	//Unbind
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}


void VBO::Draw(GLuint shaderProgramID)
{
	glBindBuffer( GL_ARRAY_BUFFER, ID );

	GLuint p = glGetAttribLocation(shaderProgramID, "vPosition");
	GLuint n = glGetAttribLocation(shaderProgramID, "vNormal");
	GLuint t = glGetAttribLocation(shaderProgramID, "vTexture");

//	//Set up Vertex Arrays  
	glEnableVertexAttribArray( p );
	glVertexAttribPointer( (GLuint)p, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0) );
	glEnableVertexAttribArray( n );
	glVertexAttribPointer( (GLuint)n, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(count * sizeof(glm::vec3)));
	glEnableVertexAttribArray( t );
	glVertexAttribPointer( (GLuint)t, 2, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(2 * count * sizeof(glm::vec3)));

	glDrawArrays(GL_TRIANGLES, 0, count);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
}