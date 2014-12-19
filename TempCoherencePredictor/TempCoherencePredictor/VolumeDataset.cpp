#include "VolumeDataset.h"

// Initializes volume by calling voxel reader and copying in values from header
void VolumeDataset::Init()
{
	VolumeProperties properties;
	voxelReader.LoadVolume(std::string(), std::string(), properties);

	memblock3D = properties.bufferAddress;
	timesteps = properties.timesteps;
	timePerFrame = properties.timePerFrame;
	numDims = properties.numDims;
	xRes = properties.xRes;
	yRes = properties.yRes;
	zRes = properties.zRes;
	bytesPerElement = properties.bytesPerElement;
	littleEndian = properties.littleEndian;
	elementType = properties.elementType;

	currentTimestep = 0;
	oldTime = clock();

	prevTexture3D = GenerateTexture();
	currTexture3D = GenerateTexture();
}


// Update for volume, at the moment used just for advancing timestep but could be for simulation or streaming etc.
void VolumeDataset::Update()
{
	if (timesteps > 1)
	{
		clock_t currentTime = clock();
		float time = (currentTime - oldTime) / (float) CLOCKS_PER_SEC;

		if (time > timePerFrame)
		{
			if (currentTimestep < timesteps - 1)
				currentTimestep++;
			else
				currentTimestep = 0;

			oldTime = currentTime;

			UpdateTexture();
		}
	}
}

void VolumeDataset::ReverseEndianness()
{
	std::vector<GLubyte> bytes;
	bytes.resize(xRes * yRes * zRes * bytesPerElement);

	for (int i=0; i<bytes.size(); i+=bytesPerElement)
	{
		for (int j=0; j<bytesPerElement; j++)
		{
			bytes[i+j] = memblock3D[i+(bytesPerElement-(1+j))];
		}
	}

	memcpy(memblock3D, &bytes[0], xRes * yRes * zRes * bytesPerElement);
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

	if (elementType == "MET_UCHAR")
		glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, xRes, yRes, zRes, 0,  GL_RED, GL_UNSIGNED_BYTE, (memblock3D + (textureSize * currentTimestep)));
	else
		std::cout << "Only working with uchar datasets at the moment" << std::endl;

	glPixelStoref(GL_UNPACK_SWAP_BYTES, false);
	
	glBindTexture(GL_TEXTURE_3D, 0);

	return tex;
}


void VolumeDataset::UpdateTexture()
{
	glDeleteTextures(1, &prevTexture3D);
	prevTexture3D = currTexture3D;

	currTexture3D = GenerateTexture();
}