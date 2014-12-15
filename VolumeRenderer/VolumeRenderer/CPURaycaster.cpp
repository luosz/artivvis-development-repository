#include "CPURaycaster.h"

CPURaycaster::CPURaycaster(int screenWidth, int screenHeight, VolumeDataset &volume)
{
	maxRaySteps = 1000;
	rayStepSize = 0.01f;
	gradientStepSize = 0.005f;
	lightPosition = glm::vec3(0.0f, -5.0f, -5.0f);

	numXPixels = screenWidth;
	numYPixels = screenHeight;

	minRange = 0.0f;
	cutOff = 0.5f;
	maxRange = 1.0f;
	
	int numOpacityDivisions = 4;

	for (int i=0; i<numOpacityDivisions; i++)
	{
		opacityDivisions.push_back(glm::vec2(0.0f, 0.0f));
		opacities.push_back(0.0f);
	}

	GenerateTexture(screenWidth, screenHeight);

//	if (!volume.littleEndian)
//		volume.ReverseEndianness();

}

void CPURaycaster::GenerateTexture(int screenWidth, int screenHeight)
{
	glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, screenWidth, screenHeight, 0, GL_RGBA, GL_FLOAT, NULL); 
	glBindTexture(GL_TEXTURE_2D, 0);
}



void CPURaycaster::Raycast(VolumeDataset &volume, TransferFunction &transferFunction, GLuint shaderProgramID, Camera &camera)
{
	float xMin, yMin, zMin, xMax, yMax, zMax;
	xMin = yMin = zMin = -1.0f;
	xMax = yMax = zMax = 1.0f;
	bool entered = false;

	glm::vec3 camDirection = glm::normalize(camera.focus - camera.position);

	glm::vec3 rightVec = glm::normalize(glm::cross(camDirection, glm::vec3(0.0f, 1.0f, 0.0f)));
	glm::vec3 upVec = glm::normalize(glm::cross(camDirection, -rightVec));

	float extent = glm::tan((camera.FoV / 2.0f) * (glm::pi<float>()/180.0f));

	glm::vec3 topLeft;

	glm::vec3 temp = camera.position + camDirection;
	temp = temp + (extent * upVec);
	topLeft = temp - (extent * rightVec);
	
	float delta = (extent * 2.0f) / (float)numXPixels;
	
	std::vector<float> scalars;
	scalars.resize(numYPixels*numXPixels*4);

	float opacity1 = opacities[0];
	float opacity2 = opacities[1];
	

	for (int y=0; y<numYPixels; y++)
	{
		for (int x=0; x<numXPixels; x++)
		{
			glm::vec3 pixelPos = glm::vec3(topLeft + ((delta * x) * rightVec) - ((delta * y) * upVec));

			glm::vec3 rayDir = glm::normalize(pixelPos - camera.position);

			entered = false;
			float absorption = 0.0f;
			bool hitDiv1 = false;
			bool hitDiv2 = false;
			glm::vec4 finalColor = glm::vec4(0.0f);
			glm::vec4 color = glm::vec4(0.0f);
			glm::vec3 normal = glm::vec3(0.0f);

			for (int i=0; i<800; i++)
			{
				glm::vec3 rayPosition = camera.position + (rayDir*(float)i*rayStepSize);
			
				if (glm::abs(rayPosition.x) < 1.0f && glm::abs(rayPosition.y) < 1.0f && glm::abs(rayPosition.z) < 1.0f && absorption < 1.0f)
				{
					entered = true;

					int discreteX = (rayPosition.x - xMin) / (2.0f / (float)volume.xRes);
					int discreteY = (rayPosition.y - yMin) / (2.0f / (float)volume.yRes);
					int discreteZ = (rayPosition.z - zMin) / (2.0f / (float)volume.zRes);

					if (discreteX < volume.xRes && discreteY < volume.yRes && discreteZ < volume.zRes)
					{
						int index = discreteX + (discreteY * volume.xRes) + (discreteZ * volume.xRes * volume.yRes);

						float value = ByteToFloat(volume, index);


						if (value > opacityDivisions[0].x && value < opacityDivisions[0].y && !hitDiv1)	
						{
							
							color = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
							normal = CalculateNormal(discreteX, discreteY, discreteZ, volume);

							color = CalculateLighting(color, normal, rayPosition);

							if ((absorption + opacity1) > 1.0f)
								color = (1.0f - absorption) * color;
							else
								color = opacity1 * color;

							finalColor += color;
							absorption += opacity1;
							
							hitDiv1 = true;
						}	
						
						else if (value > opacityDivisions[1].x && value < opacityDivisions[1].y && !hitDiv2)
						{
							color = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
							normal = CalculateNormal(discreteX, discreteY, discreteZ, volume);

							color = CalculateLighting(color, normal, rayPosition);

							if ((absorption + opacity2) > 1.0f)
								color = (1.0f - absorption) * color;
							else
								color = opacity2 * color;

							finalColor += color;
							absorption += opacity2;

							hitDiv2 = true;
						}

						scalars[((x + (y *numXPixels))*4)] = finalColor.x;
						scalars[((x + (y *numXPixels))*4)+1] = finalColor.y;
						scalars[((x + (y *numXPixels))*4)+2] = finalColor.z;
						scalars[((x + (y *numXPixels))*4)+3] = 1.0f;
					}
				}
				else
				{
					if (entered)
						break;
				}
			}
		}
	}

	glActiveTexture (GL_TEXTURE0);
	int texLoc = glGetUniformLocation(shaderProgramID,"texColor");
	glUniform1i(texLoc,0);
	glBindTexture (GL_TEXTURE_2D, texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0,0,0, numXPixels, numYPixels, GL_RGBA, GL_FLOAT, &scalars[0]);
	
	

	int texcoords_location = glGetAttribLocation (shaderProgramID, "vTexture");


	glBegin(GL_QUADS);
	glVertexAttrib2f(texcoords_location, 1.0f, 1.0f);
	glVertex2f(1.0f, 1.0f);

	glVertexAttrib2f(texcoords_location, 0.0f, 1.0f);
	glVertex2f(-1.0f, 1.0f);

	glVertexAttrib2f(texcoords_location, 0.0f, 0.0f);
	glVertex2f(-1.0f, -1.0f);

	glVertexAttrib2f(texcoords_location, 1.0f, 0.0f);
	glVertex2f(1.0f, -1.0f);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, 0);
}


glm::vec3 CPURaycaster::CalculateNormal(int x, int y, int z, VolumeDataset &volume)
{
	glm::vec3 sample1, sample2;

	if (x == 0 || y == 0 || z == 0 || x == volume.xRes-1 || y == volume.yRes-1 || z == volume.zRes-1)
		return glm::vec3(0.0f);

	int index = (x-1) + (y * volume.xRes) + (z * volume.xRes * volume.yRes);
	sample1.x = ByteToFloat(volume, index);
	index = (x+1) + (y * volume.xRes) + (z * volume.xRes * volume.yRes);
	sample2.x = ByteToFloat(volume, index);
	index = x + ((y-1) * volume.xRes) + (z * volume.xRes * volume.yRes);
	sample1.y = ByteToFloat(volume, index);
	index = x + ((y+1) * volume.xRes) + (z * volume.xRes * volume.yRes);
	sample2.y = ByteToFloat(volume, index);
	index = x + (y * volume.xRes) + ((z-1) * volume.xRes * volume.yRes);
	sample1.z = ByteToFloat(volume, index);
	index = x + (y * volume.xRes) + ((z+1) * volume.xRes * volume.yRes);
	sample2.z = ByteToFloat(volume, index);

	return glm::normalize(sample2 - sample1);
}


glm::vec4 CPURaycaster::CalculateLighting(glm::vec4 color, glm::vec3 N, glm::vec3 rayPosition)
{
	glm::vec3 lightDirection = glm::vec3(1.0f, 1.0f, 1.0f);
	glm::vec4 diffuseLight = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);
	glm::vec4 specularLight = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);

	glm::vec3 L, H;

	L = glm::normalize(lightDirection);
	H = glm::normalize(L + glm::normalize(-rayPosition));
	
	float diff = glm::clamp(glm::dot(N,L), 0.0f, 1.0f);
	glm::vec4 spec = specularLight * pow (glm::max(0.0f, glm::dot(H, N)), 50.0f); 

	return ((color * diff) + spec);
}


float CPURaycaster::ByteToFloat(VolumeDataset &volume, int index)
{
	float f;

	if (volume.elementType == "MET_UCHAR")
		f = volume.memblock3D[index] / 255.0f;

	else if (volume.elementType == "SHORT")
	{
		index *= 2;
		unsigned short s = (volume.memblock3D[index] << 8) | volume.memblock3D[(index)+1];
		f = (s / 65536.0f);
	}

	else if (volume.elementType == "FLOAT")
	{
		index *= 4;
		f = (volume.memblock3D[index] << 24) | (volume.memblock3D[(index)+1] << 16) | (volume.memblock3D[(index)+2] << 8) | volume.memblock3D[(index)+3];
	}

	return f;
}