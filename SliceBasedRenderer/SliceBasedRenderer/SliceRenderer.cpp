#include "SliceRenderer.h"

SliceRenderer::SliceRenderer(int screenWidth, int screenHeight, VolumeDataset &volume)
{
	sliceToDraw = 0;

	GenerateTextures(volume);

	transferFunction.Init(" ", volume);
}



// Generates the original 3D texture
void SliceRenderer::GenerateTextures(VolumeDataset &volume)
{
	GenerateXTextures(volume);
	GenerateYTextures(volume);
	GenerateZTextures(volume);
}


void SliceRenderer::GenerateXTextures(VolumeDataset &volume)
{
	int textureSize = volume.yRes * volume.zRes * volume.bytesPerElement;

	xSlices.reserve(volume.xRes);

	int xGap = volume.bytesPerElement;
	int yGap = volume.xRes * volume.bytesPerElement;
	int zGap = volume.xRes * volume.yRes * volume.bytesPerElement;

	std::vector<GLubyte> xBlock;

	for (int x=0; x<volume.xRes; x++)
	{
		xBlock.reserve(textureSize);

		for (int y=0; y<volume.yRes; y++)
		{
			for (int z=0; z<volume.zRes; z++)
			{
				int ID = (z * zGap) + (y * yGap) + (x * xGap);

				xBlock.push_back(volume.memblock3D[ID]);
			}
		}

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		// Reverses endianness in copy
		if (!volume.littleEndian)
			glPixelStoref(GL_UNPACK_SWAP_BYTES, true);

		if (volume.elementType == "MET_UCHAR")
			xSlices.push_back(Texture2D(GL_R8, volume.zRes, volume.yRes, GL_RED, GL_UNSIGNED_BYTE, &xBlock[0]));

		else if (volume.elementType == "SHORT")
			xSlices.push_back(Texture2D(GL_R16F, volume.zRes, volume.yRes, GL_RED, GL_UNSIGNED_SHORT, &xBlock[0]));

		else if (volume.elementType == "FLOAT")
			xSlices.push_back(Texture2D(GL_R32F, volume.zRes, volume.yRes, GL_RED, GL_FLOAT, &xBlock[0]));

		glPixelStoref(GL_UNPACK_SWAP_BYTES, false);

		xBlock.clear();
	}
}


void SliceRenderer::GenerateYTextures(VolumeDataset &volume)
{
	int textureSize = volume.xRes * volume.zRes * volume.bytesPerElement;

	ySlices.reserve(volume.yRes);

	int xGap = volume.bytesPerElement;
	int yGap = volume.xRes * volume.bytesPerElement;
	int zGap = volume.xRes * volume.yRes * volume.bytesPerElement;

	std::vector<GLubyte> yBlock;

	for (int y=0; y<volume.yRes; y++)
	{
		yBlock.reserve(textureSize);

		for (int z=0; z<volume.zRes; z++)
		{
			for (int x=0; x<volume.xRes; x++)
			{
				int ID = (z * zGap) + (y * yGap) + (x * xGap);

				yBlock.push_back(volume.memblock3D[ID]);
			}
		}

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		// Reverses endianness in copy
		if (!volume.littleEndian)
			glPixelStoref(GL_UNPACK_SWAP_BYTES, true);

		if (volume.elementType == "MET_UCHAR")
			ySlices.push_back(Texture2D(GL_R8, volume.xRes, volume.zRes, GL_RED, GL_UNSIGNED_BYTE, &yBlock[0]));

		else if (volume.elementType == "SHORT")
			ySlices.push_back(Texture2D(GL_R16F, volume.xRes, volume.zRes, GL_RED, GL_UNSIGNED_SHORT, &yBlock[0]));

		else if (volume.elementType == "FLOAT")
			ySlices.push_back(Texture2D(GL_R32F, volume.xRes, volume.zRes, GL_RED, GL_FLOAT, &yBlock[0]));

		glPixelStoref(GL_UNPACK_SWAP_BYTES, false);

		yBlock.clear();
	}
}

void SliceRenderer::GenerateZTextures(VolumeDataset &volume)
{
	int textureSize = volume.xRes * volume.yRes * volume.bytesPerElement;

	zSlices.reserve(volume.zRes);

	for (int i=0; i<volume.zRes; i++)
	{
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		// Reverses endianness in copy
		if (!volume.littleEndian)
			glPixelStoref(GL_UNPACK_SWAP_BYTES, true);

		if (volume.elementType == "MET_UCHAR")
			zSlices.push_back(Texture2D(GL_R8, volume.xRes, volume.yRes, GL_RED, GL_UNSIGNED_BYTE, volume.memblock3D + (i * textureSize)));

		else if (volume.elementType == "SHORT")
			zSlices.push_back(Texture2D(GL_R16F, volume.xRes, volume.yRes, GL_RED, GL_UNSIGNED_SHORT, volume.memblock3D + (i * textureSize)));

		else if (volume.elementType == "FLOAT")
			zSlices.push_back(Texture2D(GL_R32F, volume.xRes, volume.yRes, GL_RED, GL_FLOAT, volume.memblock3D + (i * textureSize)));

		glPixelStoref(GL_UNPACK_SWAP_BYTES, false);
	}
}



void SliceRenderer::Draw(VolumeDataset &volume, Camera &camera)
{
//	xSlices[sliceToDraw].Render();

	GLuint shaderProgramID = ShaderManager::UseShader(TFShader);

	GLint uniformLoc = glGetUniformLocation (shaderProgramID, "proj");
	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &camera.projMat[0][0]);

	uniformLoc = glGetUniformLocation (shaderProgramID, "view");
	glUniformMatrix4fv (uniformLoc, 1, GL_FALSE, &camera.viewMat[0][0]);

	glActiveTexture(GL_TEXTURE1);
	uniformLoc = glGetUniformLocation(shaderProgramID, "transferFunc");
	glUniform1i(uniformLoc, 1);
	glBindTexture(GL_TEXTURE_1D, transferFunction.tfTexture);

	GLint attribLoc = glGetAttribLocation (shaderProgramID, "vTexture");

//	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);

//	glBlendEquationEXT(GL_MAX_EXT);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//	glBlendFunc(GL_SRC_COLOR, GL_DST_COLOR);

	int axisToRender;
	float sliceGap;

	glm::vec3 viewDir = camera.GetViewDirection();

	if (glm::abs(viewDir.x) > glm::abs(viewDir.y))
	{
		if (glm::abs(viewDir.x) > glm::abs(viewDir.z))
			axisToRender = 0;
		else
			axisToRender = 2;
	}
	else
	{
		if (glm::abs(viewDir.y) > glm::abs(viewDir.z))
			axisToRender = 1;
		else
			axisToRender = 2;
	}



	if (axisToRender == 0)
	{
		sliceGap = 2.0f / volume.xRes;

		if (viewDir.x > 0.0f)
		{
			for (int i=volume.xRes-1; i>=0; i--)
				RenderXSlice(i, &xSlices[i], sliceGap, shaderProgramID, attribLoc);
		}
		else
		{
			for (int i=0; i<volume.xRes; i++)
				RenderXSlice(i, &xSlices[i], sliceGap, shaderProgramID, attribLoc);
		}
	}

	else if (axisToRender == 1)
	{
		sliceGap = 2.0f / volume.yRes;

		if (viewDir.y > 0.0f)
		{
			for (int i=volume.yRes-1; i>=0; i--)
				RenderYSlice(i, &ySlices[i], sliceGap, shaderProgramID, attribLoc);
		}
		else
		{
			for (int i=0; i<volume.yRes; i++)
				RenderYSlice(i, &ySlices[i], sliceGap, shaderProgramID, attribLoc);
		}
	}

	else
	{
		sliceGap = 2.0f / volume.zRes;

		if (viewDir.z > 0.0f)
		{
			for (int i=volume.zRes-1; i>=0; i--)
				RenderZSlice(i, &zSlices[i], sliceGap, shaderProgramID, attribLoc);
		}
		else
		{
			for (int i=0; i<volume.zRes; i++)
				RenderZSlice(i, &zSlices[i], sliceGap, shaderProgramID, attribLoc);
		}
	}

	
}


void SliceRenderer::RenderXSlice(int index, Texture2D *sliceToRender, float sliceGap, GLuint shaderProgramID, GLint attribLoc)
{
	glActiveTexture(GL_TEXTURE0);
	GLint uniformLoc = glGetUniformLocation(shaderProgramID, "slice2D");
	glUniform1i(uniformLoc, 0);
	glBindTexture(GL_TEXTURE_2D, sliceToRender->ID);

	glBegin(GL_QUADS);

	glVertexAttrib2f(attribLoc, 0.0f, 0.0f);
	glVertex3f(-1.0f + (index * sliceGap), -1.0f, -1.0f);

	glVertexAttrib2f(attribLoc, 0.0f, 1.0f);
	glVertex3f(-1.0f + (index * sliceGap), 1.0f, -1.0f);

	glVertexAttrib2f(attribLoc, 1.0f, 1.0f);
	glVertex3f(-1.0f + (index * sliceGap), 1.0f, 1.0f);

	glVertexAttrib2f(attribLoc, 1.0f, 0.0f);
	glVertex3f(-1.0f + (index * sliceGap), -1.0f, 1.0f);

	glEnd();
}

void SliceRenderer::RenderYSlice(int index, Texture2D *sliceToRender, float sliceGap, GLuint shaderProgramID, GLint attribLoc)
{
	glActiveTexture(GL_TEXTURE0);
	GLint uniformLoc = glGetUniformLocation(shaderProgramID, "slice2D");
	glUniform1i(uniformLoc, 0);
	glBindTexture(GL_TEXTURE_2D, sliceToRender->ID);

	glBegin(GL_QUADS);

	glVertexAttrib2f(attribLoc, 0.0f, 0.0f);
	glVertex3f(-1.0f, -1.0f + (index * sliceGap), -1.0f);

	glVertexAttrib2f(attribLoc, 0.0f, 1.0f);
	glVertex3f(-1.0f, -1.0f + (index * sliceGap), 1.0f);

	glVertexAttrib2f(attribLoc, 1.0f, 1.0f);
	glVertex3f(1.0f, -1.0f + (index * sliceGap), 1.0f);

	glVertexAttrib2f(attribLoc, 1.0f, 0.0f);
	glVertex3f(1.0f, -1.0f + (index * sliceGap), -1.0f);

	glEnd();
}

void SliceRenderer::RenderZSlice(int index, Texture2D *sliceToRender, float sliceGap, GLuint shaderProgramID, GLint attribLoc)
{
	glActiveTexture(GL_TEXTURE0);
	GLint uniformLoc = glGetUniformLocation(shaderProgramID, "slice2D");
	glUniform1i(uniformLoc, 0);
	glBindTexture(GL_TEXTURE_2D, sliceToRender->ID);

	glBegin(GL_QUADS);

	glVertexAttrib2f(attribLoc, 0.0f, 0.0f);
	glVertex3f(-1.0f, -1.0f, -1.0f + (index * sliceGap));

	glVertexAttrib2f(attribLoc, 0.0f, 1.0f);
	glVertex3f(-1.0f, 1.0f, -1.0f + (index * sliceGap));

	glVertexAttrib2f(attribLoc, 1.0f, 1.0f);
	glVertex3f(1.0f, 1.0f, -1.0f + (index * sliceGap));

	glVertexAttrib2f(attribLoc, 1.0f, 0.0f);
	glVertex3f(1.0f, -1.0f, -1.0f + (index * sliceGap));

	glEnd();
}