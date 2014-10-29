#include "BurnsContours.h"

void BurnsContours::Init(int screenWidth, int screenHeight, VolumeDataset &volume)
{
	compute = false;

	glGenFramebuffers (1, &frameBuffer);
	glBindFramebuffer (GL_FRAMEBUFFER, frameBuffer);

	unsigned int rb = 0;
	glGenRenderbuffers (1, &rb);
	glBindRenderbuffer (GL_RENDERBUFFER, rb);
	glRenderbufferStorage (GL_RENDERBUFFER, GL_DEPTH_COMPONENT, screenWidth, screenHeight);
	glFramebufferRenderbuffer (GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb);

	glBindFramebuffer (GL_FRAMEBUFFER, 0);



	glGenTextures(1, &depthTexture);
	glBindTexture (GL_TEXTURE_2D, depthTexture);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA32F, screenWidth, screenHeight, 0, GL_RGBA, GL_FLOAT, NULL);

	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, depthTexture, 0);


	glGenTextures(1, &contoursTexture);
	glBindTexture (GL_TEXTURE_2D, contoursTexture);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA32F, screenWidth, screenHeight, 0, GL_RGBA, GL_FLOAT, NULL);

	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, contoursTexture, 0);


	CalculateNormals(volume);
}

void BurnsContours::CalculateNormals(VolumeDataset &volume)
{
	int xRes = volume.xRes;
	int yRes = volume.yRes;
	int zRes = volume.zRes;

	normals.reserve(xRes*yRes*zRes);

	int index;
	int idx1, idx2;
	glm::vec3 sample1, sample2;
	float dist;

	for (int k=0; k<zRes; k++)
	{
		for (int j=0; j<yRes; j++)
		{
			for (int i=0; i<xRes; i++)
			{
				sample1 = glm::vec3(0.0f);
				sample2 = glm::vec3(0.0f);

				index = i + (j * xRes) + (k * xRes * yRes);
				idx1 = (i-1) + ((j-1) * xRes) + ((k-1) * xRes * yRes);
				idx2 = (i+1) + ((j+1) * xRes) + ((k+1) * xRes * yRes);

				if (i % xRes != 0)
					sample1.x = volume.memblock3D[((i-1) + (j * xRes) + (k * xRes * yRes)) * volume.bytesPerElement];
				if (j % yRes != 0)
					sample1.y = volume.memblock3D[(i + ((j-1) * xRes) + (k * xRes * yRes)) * volume.bytesPerElement];
				if (k % zRes != 0)
					sample1.z = volume.memblock3D[(i + (j * xRes) + ((k-1) * xRes * yRes)) * volume.bytesPerElement];

				if (i % xRes != xRes-1)
					sample2.x = volume.memblock3D[((i+1) + (j * xRes) + (k * xRes * yRes)) * volume.bytesPerElement];
				if (j % yRes != yRes-1)
					sample2.y = volume.memblock3D[(i + ((j+1) * xRes) + (k * xRes * yRes)) * volume.bytesPerElement];
				if (k % zRes != zRes-1)
					sample2.z = volume.memblock3D[(i + (j * xRes) + ((k+1) * xRes * yRes)) * volume.bytesPerElement];

				dist = glm::distance(sample1, sample2);

				if (dist == 0.0f)
					normals.push_back(glm::vec3(0.0f));
				else
					normals.push_back((sample1 - sample2) / dist);
			}
		}
	}

}


void BurnsContours::DrawContours(VolumeDataset &volume, Camera &camera, ShaderManager &shaderManager, Raycaster &raycaster)
{
	if (compute)
	{
		CalculateContours(volume, camera, raycaster);
		compute = false;
	}

	Display(volume, camera, raycaster, shaderManager);
}



glm::vec4 BurnsContours::INTER(float isoValue, glm::vec3 p1, glm::vec3 p2, float valp1, float valp2)
{
	glm::vec4 p = glm::vec4(0.0f);
	float mu;
	
	if ((valp1 < isoValue && valp2 >= isoValue) || (valp1 >= isoValue && valp2 < isoValue))
	{
		mu = (isoValue - valp1) / (valp2 - valp1);
		p.x = p1.x + mu * (p2.x - p1.x);
		p.y = p1.y + mu * (p2.y - p1.y);
		p.z = p1.z + mu * (p2.z - p1.z);

		if (valp2 >= isoValue)
			p.w = 1.0f;
		else 
			p.w = -1.0f;
	}

	return p;
}





void BurnsContours::SEGMENTS(glm::vec4* intersections, float* valAtIntersection, int numIntersections, Segment* segments, int &numSegments)
{
	if (numIntersections == 0)
		return;

	if (numIntersections >= 2)
	{
		if (intersections[0].w < 0.0f)
			segments[numSegments] = Segment(glm::vec3(intersections[0]), glm::vec3(intersections[1]), valAtIntersection[0], valAtIntersection[1]);
		else
			segments[numSegments] = Segment(glm::vec3(intersections[1]), glm::vec3(intersections[0]), valAtIntersection[1], valAtIntersection[0]);

		numSegments += 1;
	}

	if (numIntersections == 4)
	{
		if (intersections[2].w < 0.0f)
			segments[numSegments] = Segment(glm::vec3(intersections[2]), glm::vec3(intersections[3]), valAtIntersection[2], valAtIntersection[3]);
		else
			segments[numSegments] = Segment(glm::vec3(intersections[3]), glm::vec3(intersections[2]), valAtIntersection[3], valAtIntersection[2]);

		numSegments += 1;
	}
}


float BurnsContours::LERPfloat(glm::vec3 intersectionPoint, glm::vec3 p1, glm::vec3 p2, float valp1, float valp2)
{
	float dist = glm::distance(p1, p2);

	float scalar = glm::distance(intersectionPoint, p1) / dist;

	float value = (valp1 + ((valp2 - valp1) * scalar));

	return (valp1 + ((valp2 - valp1) * scalar));

}

glm::vec3 BurnsContours::LERPvec(glm::vec3 intersectionPoint, glm::vec3 p1, glm::vec3 p2, glm::vec3 valp1, glm::vec3 valp2)
{
	float dist = glm::distance(p1, p2);

	float scalar = glm::distance(intersectionPoint, p1) / dist;

	glm::vec3 value = (valp1 + ((valp2 - valp1) * scalar));

	return (valp1 + ((valp2 - valp1) * scalar));
}


void BurnsContours::CalculateContours(VolumeDataset &volume, Camera &camera, Raycaster &raycaster)
{
	glm::vec3 cellPosition;

	float cubeMin = -1.0f;
	float cubeExtent = 2.0f;
	glm::vec3 V;

	vertices.clear();
	colors.clear();

	int xRes = volume.xRes;
	int yRes = volume.yRes;
	int zRes = volume.zRes;

	

	float isoValue = ((raycaster.opacityDivisions[1].y + raycaster.opacityDivisions[1].x) / 2.0f) * 255;
	float isoValue2 = 0.0f;

	for (int k=1; k<zRes; k++)
	{
		for (int j=1; j<yRes; j++)
		{
			for (int i=1; i<xRes; i++)
			{
				cellPosition = cubeMin + glm::vec3((i * (cubeExtent / (float)xRes)), (j * (cubeExtent / (float)yRes)), (k * (cubeExtent / (float)zRes)));

				float cornerValues[8];
				float cornerValues2[8];
				int cornerIndices[8];
				glm::vec3 cornerLocations[8];

				for (int l=0; l<8; l++)
				{
					cornerLocations[l].x = vertexConnections[l][0];
					cornerLocations[l].y = vertexConnections[l][1];
					cornerLocations[l].z = vertexConnections[l][2];
					cornerIndices[l] = ((i + cornerLocations[l].x) + ((j + cornerLocations[l].y) * xRes) + ((k + cornerLocations[l].z) * xRes * yRes));

					cornerValues[l] = volume.memblock3D[cornerIndices[l] * volume.bytesPerElement];	



					glm::vec3 temp = cubeMin + glm::vec3(((i+vertexConnections[l][0]) * (cubeExtent / (float)xRes)), ((j+vertexConnections[l][1]) * (cubeExtent / (float)yRes)), ((k+vertexConnections[l][2]) * (cubeExtent / (float)zRes)));
					V = glm::normalize(temp - camera.position);
					cornerValues2[l] = glm::dot(normals[cornerIndices[l]], V);

				}

				Segment segments[12];
				int numSegments = 0;

				int numIntersections = 0;
				glm::vec4 intersectionPoints[4];
				float val2AtIntersection[4];		

				for (int l=0; l<6; l++)
				{
					numIntersections = 0;

					for (int k=0; k<4; k++)
					{

						glm::vec4 temp = INTER(isoValue, cornerLocations[faceConnections[(l*4) + k][0]], cornerLocations[faceConnections[(l*4) + k][1]], cornerValues[faceConnections[(l*4) + k][0]], cornerValues[faceConnections[(l*4) + k][1]]);
						
						if (temp.w != 0.0f)
						{
							intersectionPoints[numIntersections] = temp;

							val2AtIntersection[numIntersections] = LERPfloat(glm::vec3(temp), cornerLocations[faceConnections[(l*4) + k][0]], cornerLocations[faceConnections[(l*4) + k][1]], cornerValues2[faceConnections[(l*4) + k][0]], cornerValues2[faceConnections[(l*4) + k][1]]);

							numIntersections++;
						}
					}
					SEGMENTS(&intersectionPoints[0], &val2AtIntersection[0], numIntersections, &segments[0], numSegments);
				}


				for (int l=0; l<numSegments; l++)
				{
					glm::vec4 point = INTER(isoValue2, segments[l].s1, segments[l].s2, segments[l].g1, segments[l].g2);

					if (point.w != 0.0f)
					{
						colors.push_back(normals[cornerIndices[0]]);
						vertices.push_back(glm::vec3(cellPosition.x + (point.x * (cubeExtent / (float)xRes)), cellPosition.y + (point.y * (cubeExtent / (float)yRes)), cellPosition.z + (point.z * (cubeExtent / (float)zRes))));
					}
				}
			}
		}
	}
}


void BurnsContours::Display(VolumeDataset &volume, Camera &camera, Raycaster &raycaster, ShaderManager &shaderManager)
{
//	GLuint shaderProgramID = shaderManager.UseShader(DepthShader);
//	
//
//	glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
//	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, depthTexture, 0);
//	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
//	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//	
//	raycaster.Raycast(volume, NULL, shaderProgramID, camera);
//
//
//
//	glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, contoursTexture, 0);
//	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
//	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//	
//	shaderProgramID = shaderManager.UseShader(LineDepthShader);
//	
//
//	int uniformLoc;
//	glm::mat4 model_mat = glm::mat4(1.0f);
//
//	uniformLoc = glGetUniformLocation (shaderProgramID, "proj");
//	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &camera.projMat[0][0]);
//
//	uniformLoc = glGetUniformLocation (shaderProgramID, "view");
//	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &camera.viewMat[0][0]);
//
//	uniformLoc = glGetUniformLocation (shaderProgramID, "model");
//	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &model_mat[0][0]);
//
//	uniformLoc = glGetUniformLocation(shaderProgramID,"camPos");
//	glUniform3f(uniformLoc, camera.position.x, camera.position.y, camera.position.z);
//
//
//	glBegin(GL_LINES);
//
//	for (int i=0; i<vertices.size(); i++)
//	{
//		glColor3f(colors[i].x, colors[i].y, colors[i].z);
//		glVertex3f(vertices[i].x, vertices[i].y, vertices[i].z);
//	}
//
//	glEnd();
//
//	
//	shaderProgramID = shaderManager.UseShader(ComparisonShader);
//	glBindFramebuffer (GL_FRAMEBUFFER, 0);
//
//	
//	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
//
//	glActiveTexture (GL_TEXTURE0);
//	 int uniformLocation = glGetUniformLocation(shaderProgramID,"depthImage");
//	glUniform1i(uniformLocation, 0);
//	glBindTexture (GL_TEXTURE_2D, depthTexture);
//
//	glActiveTexture (GL_TEXTURE1);
//	uniformLocation = glGetUniformLocation(shaderProgramID,"contoursImage");
//	glUniform1i(uniformLocation, 1);
//	glBindTexture (GL_TEXTURE_2D, contoursTexture);
//
//
//
//	int texcoords_location = glGetAttribLocation (shaderProgramID, "vTexture");
//	glBegin(GL_QUADS);
//	glVertexAttrib2f(texcoords_location, 0.0f, 0.0f);
//	glVertex2f(-1.0f, -1.0f);
//
//	glVertexAttrib2f(texcoords_location, 1.0f, 0.0f);
//	glVertex2f(1.0f, -1.0f);
//
//	glVertexAttrib2f(texcoords_location, 1.0f, 1.0f);
//	glVertex2f(1.0f, 1.0f);
//
//	glVertexAttrib2f(texcoords_location, 0.0f, 1.0f);
//	glVertex2f(-1.0f, 1.0f);
//	glEnd();
}





const int BurnsContours::vertexConnections[8][3] = {
	{0, 0, 0},{-1, 0, 0},{-1, -1, 0},{0, -1, 0},
	{0, 0, -1},{-1, 0, -1},{-1, -1, -1},{0, -1, -1},
};


const int BurnsContours::faceConnections[24][2] = {
        {0,1}, {1,2}, {2,3}, {3,0},
        {7,6}, {6,5}, {5,4}, {4,7},
		{4,5}, {5,1}, {1,0}, {0,4},
		{3,2}, {2,6}, {6,7}, {7,3},
		{4,0}, {0,3}, {3,7}, {7,4},
		{1,5}, {5,6}, {6,2}, {2,1}
};
