#include "VolumeDataset.h"

// Initializes volume by calling voxel reader and copying in values from header
void VolumeDataset::Init()
{
	numVoxels = xRes * yRes * zRes;

	int bufferSize = xRes * yRes * zRes * bytesPerElement;
	memblock3D = new GLubyte [bufferSize];

	std::memset(memblock3D, 0, bufferSize);
	currTexture3D = GenerateTexture();
}

void VolumeDataset::CopyToTexture()
{
	glBindTexture(GL_TEXTURE_3D, currTexture3D);

	if (!littleEndian)
		glPixelStoref(GL_UNPACK_SWAP_BYTES, true);

	if (bytesPerElement == 1)
		glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, xRes, yRes, zRes, GL_RED, GL_UNSIGNED_BYTE, memblock3D);

	else if (bytesPerElement == 2)
		glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, xRes, yRes, zRes, GL_RED, GL_UNSIGNED_SHORT, memblock3D);

	else if (bytesPerElement == 4)
		glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, xRes, yRes, zRes, GL_RED, GL_FLOAT, memblock3D);

	glPixelStoref(GL_UNPACK_SWAP_BYTES, false);

	glBindTexture(GL_TEXTURE_3D, 0);
}

GLuint VolumeDataset::GenerateTexture()
{
	GLuint tex;
	int textureSize = xRes * yRes * zRes * bytesPerElement;

	glEnable(GL_TEXTURE_3D);
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_3D, tex);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);


	// Reverses endianness in copy
	if (!littleEndian)
		glPixelStoref(GL_UNPACK_SWAP_BYTES, true);

	if (bytesPerElement == 1)
		glTexImage3D(GL_TEXTURE_3D, 0, GL_R8,xRes, yRes, zRes, 0,  GL_RED, GL_UNSIGNED_BYTE, memblock3D);

	else if (bytesPerElement == 2)
		glTexImage3D(GL_TEXTURE_3D, 0, GL_R16F, xRes, yRes, zRes, 0, GL_RED, GL_UNSIGNED_SHORT, memblock3D);

	else if (bytesPerElement == 4)
		glTexImage3D(GL_TEXTURE_3D, 0, GL_R16F, xRes, yRes, zRes, 0, GL_RED, GL_FLOAT, memblock3D);

	glPixelStoref(GL_UNPACK_SWAP_BYTES, false);
	
	glBindTexture(GL_TEXTURE_3D, 0);

	return tex;
}




