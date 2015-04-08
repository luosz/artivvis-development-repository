#include "VolumeDataset.h"

// Initializes volume by calling voxel reader and copying in values from header
void VolumeDataset::Init()
{
	VolumeProperties properties;
	voxelReader.LoadVolume(folderPath, headerFile, properties);

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

// Generate the 3D texture and launch the first background thread to load the next timestep
void VolumeDataset::InitTexture()
{
	currTexture3D = GenerateTexture();

	if (timesteps > 1)
		asyncTexLoad = std::async(&VolumeDataset::LoadTextureAsync, this);
}


// Update for volume, at the moment used just for advancing timestep and streaming in the next block but could be for simulation etc.
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


// Use if need to reverse endianness of CPU memblock
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


// Generates 3D texture on GPU
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


// Run in a background thread to load next timestep from disk
void VolumeDataset::LoadTextureAsync()
{
	if (currentTimestep < timesteps-1)
		voxelReader.CopyFileToBuffer(memblock3D, currentTimestep+1);
	else
		voxelReader.CopyFileToBuffer(memblock3D, 0);

	return;
}


// Copies the block corresponding to the current timestep to the GPU
void VolumeDataset::CopyToTexture()
{
	glBindTexture(GL_TEXTURE_3D, currTexture3D);

	if (!littleEndian)
		glPixelStoref(GL_UNPACK_SWAP_BYTES, true);

	if (elementType == "MET_UCHAR")
		glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, xRes, yRes, zRes, GL_RED, GL_UNSIGNED_BYTE, memblock3D);

	else if (elementType == "SHORT")
		glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, xRes, yRes, zRes, GL_RED, GL_UNSIGNED_SHORT, memblock3D);

	else if (elementType == "FLOAT")
		glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, xRes, yRes, zRes, GL_RED, GL_FLOAT, memblock3D);

	glPixelStoref(GL_UNPACK_SWAP_BYTES, false);

	glBindTexture(GL_TEXTURE_3D, 0);
}


// Waits for thread loading next timestep to complete then copies data to GPU texture
void VolumeDataset::UpdateTexture()
{
	asyncTexLoad.wait();

	CopyToTexture();
	
	asyncTexLoad = std::async(&VolumeDataset::LoadTextureAsync, this);	
}


// Read mhd filename from command-line argument in argv[1] and extract folder path from mhd filename
void VolumeDataset::ParseArguments(int argc, char *argv[])
{
	// Read filename from command-line argument argv[1] if available
	if (argc >= 2)
	{
		char filename[MAX_PATH];
		strcpy(filename, argv[1]);
		// Try both Windows and Linux directory separators ('\\' and '/')
		char *p = strrchr(filename, '\\');
		if (!p)
		{
			p = strrchr(filename, '/');
		}
		if (p)
		{
			headerFile = std::string(filename);
			if (strlen(p) >= 2)
			{
				// Extract folder path from volume filename
				p[1] = '\0';
			}
			folderPath = std::string(filename);
			std::cout << "headerFile=" << headerFile << std::endl << "folderPath=" << folderPath << std::endl;
		}
	}
}