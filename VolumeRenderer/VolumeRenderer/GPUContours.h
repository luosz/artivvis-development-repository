#ifndef GPU_CONTOURS_H
#define GPU_CONTOURS_H

#include "ContourDrawer.h"

class GPUContours		:		public ContourDrawer
{
public:
	GLuint frameBuffer;
	GLuint diffuseTexture, opacityTexture;

	void Init(int screenWidth, int screenHeight, VolumeDataset &volume);
	void DrawContours(VolumeDataset &volume, Camera &camera, ShaderManager &shaderManager, Raycaster &raycaster);
};


#endif