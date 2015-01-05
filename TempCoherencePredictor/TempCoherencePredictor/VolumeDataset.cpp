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

	numVoxels = xRes * yRes * zRes;
}


// Update for volume, at the moment used just for advancing timestep but could be for simulation or streaming etc.



