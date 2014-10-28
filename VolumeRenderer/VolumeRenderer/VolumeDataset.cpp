#include "VolumeDataset.h"

// Initializes volume by calling voxel reader and copying in values from header
void VolumeDataset::Init()
{
	VolumeProperties properties;
	VoxelReader reader;
	reader.LoadVolume(std::string(), std::string(), properties);

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
}


// Update for volume, at the moment used just for advancing timestep but could be for simulation or streaming etc.
void VolumeDataset::Update()
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
