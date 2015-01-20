#ifndef VOLUME_DATASET_H
#define VOLUME_DATASET_H

#include <string.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "VoxelReader.h"
#include <string>
#include <time.h>
#include "GLM.h"
#include "Camera.h"

#ifndef MAX_PATH
#define MAX_PATH          260
#endif

class VolumeDataset
{
public:
	GLubyte *memblock3D;
	GLuint currTexture3D;
	GLuint nextTexture3D;

	int timesteps;
	float timePerFrame;
	int numDims;
	int xRes, yRes, zRes;
	int bytesPerElement;
	bool littleEndian;
	std::string elementType;

	int currentTimestep;
	clock_t oldTime;

	VoxelReader voxelReader;

	void Init();
	void Update();
	void ReverseEndianness();
	GLuint GenerateTexture();
	void UpdateTexture();

	//////////////////////////////////////////////////////////////////////////
	// parse command-line arguments to get volume filename
	std::string folderPath;
	std::string headerFile;

	void ParseArguments(int argc, char *argv[])
	{
		std::cout << "argc=" << argc << " argv[0]=" << argv[0] << std::endl;
		// read filename from arguments if available
		if (argc >= 2)
		{
			char volume_filename[MAX_PATH];
			strcpy(volume_filename, argv[1]);
			char *p = strrchr(volume_filename, '\\');
			// try Windows and Linux directory separators ('\\' and '/')
			if (!p)
			{
				p = strrchr(volume_filename, '/');
			}
			if (p)
			{
				headerFile = std::string(volume_filename);
				std::cout << strlen(p) << std::endl;
				if (strlen(p) >= 2)
				{
					// remove filename and keep folder path
					p[1] = '\0';
				}
				folderPath = std::string(volume_filename);
				std::cout << headerFile << std::endl << folderPath << std::endl;
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////
};



#endif