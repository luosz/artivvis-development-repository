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
#include <future>

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
	std::future<GLuint> asyncTexLoad;

	VoxelReader voxelReader;

	void Init();
	void Update();
	void ReverseEndianness();
	GLuint GenerateTexture();
	void UpdateTexture();
	GLuint LoadTextureAsync();

	//////////////////////////////////////////////////////////////////////////
	std::string folderPath;
	std::string headerFile;

	/// Read mhd filename from command-line argument in argv[1] and extract folder path from mhd filename
	void ParseArguments(int argc, char *argv[])
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
	//////////////////////////////////////////////////////////////////////////
};

#endif
