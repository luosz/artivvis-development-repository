#ifndef SLICE_RENDERER_H
#define SLICE_RENDERER_H

#include "VolumeDataset.h"
#include "ShaderManager.h"
#include "TransferFunction.h"
#include "Texture2D.h"

class SliceRenderer
{
public:
	TransferFunction transferFunction;
	std::vector<Texture2D> xSlices, ySlices, zSlices;

	int sliceToDraw;

	SliceRenderer(int screenWidth, int screenHeight, VolumeDataset &volume);

	void GenerateTextures(VolumeDataset &volume);
	void GenerateXTextures(VolumeDataset &volume);
	void GenerateYTextures(VolumeDataset &volume);
	void GenerateZTextures(VolumeDataset &volume);

	void Draw(VolumeDataset &volume, Camera &camera);

	void RenderXSlice(int index, Texture2D *sliceToRender, float sliceGap, GLuint shaderProgramID, GLint attribLoc);
	void RenderYSlice(int index, Texture2D *sliceToRender, float sliceGap, GLuint shaderProgramID, GLint attribLoc);
	void RenderZSlice(int index, Texture2D *sliceToRender, float sliceGap, GLuint shaderProgramID, GLint attribLoc);
};

#endif