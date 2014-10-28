#include "StaticVolume.h"

//Reads binary files in form of 16-bit shorts and puts into single block
void StaticVolume::ReadFiles()
{
	xRes = X_PIXELS;
	yRes = Y_PIXELS;
	zRes = NUM_SLICES;

	std::streampos size;
	int bytesPerFrame = X_PIXELS * Y_PIXELS * sizeof(short);
	memblock3D = new char [bytesPerFrame * NUM_SLICES];

	for (int i=1; i<=NUM_SLICES; i++)
	{
		std::string fileName(FILE_PATH + std::to_string(i));

		std::ifstream file (fileName, std::ios::in|std::ios::binary|std::ios::ate);

		if (file.is_open())
		{
		  size = file.tellg();
		  
		  file.seekg (0, std::ios::beg);
		  file.read (memblock3D + ((i-1) * bytesPerFrame), size);
		  file.close();
		
		}
		else 
			std::cout << "Unable to open file";
	}

	
	ordered = new char [bytesPerFrame * NUM_SLICES];

	for (int i=0; i<bytesPerFrame * NUM_SLICES; i+=2)
	{
		ordered[i] = memblock3D[i+1];
		ordered[i+1] = memblock3D[i];
	}

	shorts.resize(bytesPerFrame * NUM_SLICES);
	memcpy(&shorts[0], ordered, bytesPerFrame * NUM_SLICES);
}