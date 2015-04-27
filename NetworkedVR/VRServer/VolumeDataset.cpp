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

	currentTimestep = 0;
	oldTime = clock();

	InitTexture();
}

void VolumeDataset::InitTexture()
{
	currTexture3D = GenerateTexture();

	int bufferSize = xRes * yRes * zRes * bytesPerElement;

	for (int i=0; i<NUM_STREAMING_THREADS; i++)
		threadBlock[i] = new GLubyte[bufferSize];

	if (timesteps > 1)
	{
		asyncTexLoad[0] = std::async(&VolumeDataset::LoadTextureAsync, this, 0, NUM_STREAMING_THREADS);

		for (int i=1; i<NUM_STREAMING_THREADS; i++)
			asyncTexLoad[i] = std::async(&VolumeDataset::LoadTextureAsync, this, i, i);
	}
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
		glTexImage3D(GL_TEXTURE_3D, 0, GL_R8,xRes, yRes, zRes, 0,  GL_RED, GL_UNSIGNED_BYTE, memblock3D);

	else if (elementType == "SHORT")
		glTexImage3D(GL_TEXTURE_3D, 0, GL_R16F, xRes, yRes, zRes, 0, GL_RED, GL_UNSIGNED_SHORT, memblock3D);

	else if (elementType == "FLOAT")
		glTexImage3D(GL_TEXTURE_3D, 0, GL_R16F, xRes, yRes, zRes, 0, GL_RED, GL_FLOAT, memblock3D);

	glPixelStoref(GL_UNPACK_SWAP_BYTES, false);
	
	glBindTexture(GL_TEXTURE_3D, 0);

	return tex;
}

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


// Run in a background thread to load next timestep from disk
void VolumeDataset::LoadTextureAsync(int currentThread, int stepToBuffer)
{
	voxelReader.CopyFileToBuffer(threadBlock[currentThread], stepToBuffer);

	return;
}


// Copies the block corresponding to the current timestep to the GPU
void VolumeDataset::CopyToTexture()
{
	glBindTexture(GL_TEXTURE_3D, currTexture3D);

	if (!littleEndian)
		glPixelStoref(GL_UNPACK_SWAP_BYTES, true);

	if (elementType == "MET_UCHAR")
		glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, xRes, yRes, zRes, GL_RED, GL_UNSIGNED_BYTE, currMemblock);

	else if (elementType == "SHORT")
		glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, xRes, yRes, zRes, GL_RED, GL_UNSIGNED_SHORT, currMemblock);

	else if (elementType == "FLOAT")
		glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, xRes, yRes, zRes, GL_RED, GL_FLOAT, currMemblock);

	glPixelStoref(GL_UNPACK_SWAP_BYTES, false);

	glBindTexture(GL_TEXTURE_3D, 0);
}


// Waits for thread loading next timestep to complete then copies data to GPU texture
void VolumeDataset::UpdateTexture()
{
	int currentThread = currentTimestep % NUM_STREAMING_THREADS;
	asyncTexLoad[currentThread].wait();
	currMemblock = threadBlock[currentThread];

	CopyToTexture();

	int stepToBuffer;

	if (currentTimestep + NUM_STREAMING_THREADS < timesteps)
		stepToBuffer = currentTimestep + NUM_STREAMING_THREADS;
	else
		stepToBuffer = currentThread;//(currentTimestep + NUM_STREAMING_THREADS) % timesteps;

	asyncTexLoad[currentThread] = std::async(&VolumeDataset::LoadTextureAsync, this, currentThread, stepToBuffer);	
}