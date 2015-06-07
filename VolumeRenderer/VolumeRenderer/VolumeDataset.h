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

#define NUM_STREAMING_THREADS 2

class VolumeDataset
{
public:
	GLubyte *memblock3D;
	GLubyte *threadBlock[NUM_STREAMING_THREADS];
	GLubyte *currMemblock;
	GLuint currTexture3D;
	GLuint rTexture3D;
	GLuint gTexture3D;
	GLuint bTexture3D;

	std::string folderPath;
	std::string headerFile;

	int timesteps;
	float timePerFrame;
	int numDims;
	int xRes, yRes, zRes;
	int bytesPerElement;
	bool littleEndian;
	std::string elementType;

	int currentTimestep;
	clock_t oldTime;
	std::future<void> asyncTexLoad[NUM_STREAMING_THREADS];

	VoxelReader voxelReader;

	void Init();
	void Update();
	void ReverseEndianness();
	void InitTexture();
	GLuint GenerateTexture();
	void UpdateTexture();
	void LoadTextureAsync(int currentThread, int stepsToBuffer);
	void CopyToTexture();

	void ParseArguments(int argc, char *argv[]);

	// Generates 3D texture on GPU
	GLuint generate_texture(GLuint &texture3D, GLubyte *raw)
	{
		//GLuint tex;
		int textureSize = xRes * yRes * zRes * bytesPerElement;

		glEnable(GL_TEXTURE_3D);
		glGenTextures(1, &texture3D);
		glBindTexture(GL_TEXTURE_3D, texture3D);
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
			glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, xRes, yRes, zRes, 0, GL_RED, GL_UNSIGNED_BYTE, raw);

		else if (elementType == "SHORT")
			glTexImage3D(GL_TEXTURE_3D, 0, GL_R16F, xRes, yRes, zRes, 0, GL_RED, GL_UNSIGNED_SHORT, raw);

		else if (elementType == "FLOAT")
			glTexImage3D(GL_TEXTURE_3D, 0, GL_R16F, xRes, yRes, zRes, 0, GL_RED, GL_FLOAT, raw);

		glPixelStoref(GL_UNPACK_SWAP_BYTES, false);

		glBindTexture(GL_TEXTURE_3D, 0);

		return texture3D;
	}
};

#endif
