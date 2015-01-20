#include "BruteForce.h"


BruteForce::BruteForce(VolumeDataset &volume)
{
	tex3D = GenerateTexture(volume);
}

GLuint BruteForce::GenerateTexture(VolumeDataset &volume)
{
	GLuint tex;

	textureSize = volume.xRes * volume.yRes * volume.zRes * volume.bytesPerElement;

	glEnable(GL_TEXTURE_3D);
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_3D, tex);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, volume.xRes, volume.yRes, volume.zRes, 0,  GL_RED, GL_UNSIGNED_BYTE, (volume.memblock3D));

	glBindTexture(GL_TEXTURE_3D, 0);

	return tex;
}

GLuint BruteForce::BruteForceCopy(VolumeDataset &volume, int currentTimestep)
{
	glBindTexture(GL_TEXTURE_3D, tex3D);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, volume.xRes, volume.yRes, volume.zRes, 0,  GL_RED, GL_UNSIGNED_BYTE, (volume.memblock3D + (textureSize * currentTimestep)));
	glBindTexture(GL_TEXTURE_3D, 0);

	return tex3D;
}