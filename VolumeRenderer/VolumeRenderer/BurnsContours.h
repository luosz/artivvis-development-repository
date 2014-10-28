#ifndef BURNS_CONTOURS_H
#define BURNS_CONTOURS_H

#include "ContourDrawer.h"

class BurnsContours		:		public ContourDrawer
{
public:
	GLuint frameBuffer;
	GLuint contoursTexture, depthTexture;

	std::vector<glm::vec3> vertices;
	std::vector<glm::vec3> normals;
	std::vector<glm::vec3> colors;

	void Init(int screenWidth, int screenHeight, VolumeDataset &volume);
	void DrawContours(VolumeDataset &volume, Camera &camera, ShaderManager &shaderManager, Raycaster &raycaster);

	void CalculateNormals(VolumeDataset &volume);
	void CalculateContours(VolumeDataset &volume, Camera &camera, Raycaster &raycaster);
	void Display(VolumeDataset &volume, Camera &camera, Raycaster &raycaster, ShaderManager &shaderManager);

	struct Segment
	{
		glm::vec3 s1;
		glm::vec3 s2;
	
		float g1;
		float g2;
	
		Segment()
		{
		}
	
		Segment(glm::vec3 s1_, glm::vec3 s2_, float g1_, float g2_)
		{
			s1 = s1_;
			s2 = s2_;
	
			g1 = g1_;
			g2 = g2_;
		}
	};

	glm::vec4 INTER(float isoValue, glm::vec3 p1, glm::vec3 p2, float valp1, float valp2);
	void SEGMENTS(glm::vec4* intersections, float* valAtIntersection, int numIntersections, Segment* segments, int &numSegments);
	float LERPfloat(glm::vec3 intersectionPoint, glm::vec3 p1, glm::vec3 p2, float valp1, float valp2);
	glm::vec3 LERPvec(glm::vec3 intersectionPoint, glm::vec3 p1, glm::vec3 p2, glm::vec3 valp1, glm::vec3 valp2);

	static const int vertexConnections[8][3];
	static const int faceConnections[24][2];
};


#endif