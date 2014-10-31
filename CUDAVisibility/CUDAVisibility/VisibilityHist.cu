#include "VisibilityHist.h"

void VisibilityHistogram::Init(int screenWidth, int screenHeight, VolumeDataset &volume)
{
	HANDLE_ERROR( cudaMalloc((void**)&cudaTexture, volume.xRes * volume.yRes * volume.zRes * volume.bytesPerElement) );
	HANDLE_ERROR( cudaMemcpy(cudaTexture, volume.memblock3D, volume.xRes * volume.yRes * volume.zRes * volume.bytesPerElement, cudaMemcpyHostToDevice) );

	xPixels = screenWidth;
	yPixels = screenHeight;

}



void VisibilityHistogram::CalculateHistogram(VolumeDataset &volume, TransferFunction &transferFunction, Camera &camera, ShaderManager shaderManager)
{



}