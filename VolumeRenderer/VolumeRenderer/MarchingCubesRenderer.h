#ifndef MARCHING_CUBES_RENDERER_H
#define MARCHING_CUBES_RENDERER_H

#include "Renderer.h"

class MarchingCubesRenderer		:		public Renderer
{
public:

	std::vector<glm::vec3> normals;
	std::vector<glm::vec3> surfaceVertices;
	std::vector<glm::vec3> surfaceNormals;

	void Init(int screenWidth, int screenHeight, VolumeDataset &volume);
	void Draw(VolumeDataset &volume, ShaderManager &shaderManager, Camera &camera);

	void CalculateNormals(VolumeDataset &volume);
	void CalculateSurface(VolumeDataset &volume);

	glm::vec3 LERPvec(glm::vec3 intersectionPoint, glm::vec3 p1, glm::vec3 p2, glm::vec3 valp1, glm::vec3 valp2);
	glm::vec3 VertexInterp(float isoValue, glm::vec3 p1, glm::vec3 p2, float valp1, float valp2);

	static const int vertexConnections[8][3];
	static const int edgeConnections[12][2];
	static const int edgeTable[256];
	static const int triTable[256][16];
};

#endif