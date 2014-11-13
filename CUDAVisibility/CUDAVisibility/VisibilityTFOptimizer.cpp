#include "VisibilityTFOptimizer.h"

VisibilityTFOptimizer::VisibilityTFOptimizer(VolumeDataset &volume_, int numIntensities_, glm::vec4 *colors_, float *intensities_)
{
	volume = &volume_;
	numIntensities = numIntensities_;
	colors = colors_;
	intensities= intensities_;

	numVoxels = volume->xRes * volume->yRes * volume->zRes;
}

void VisibilityTFOptimizer::Optimize(float targetIntensity, VisibilityHistogram histogram)
{
//	std::fill(frequencies.begin(), frequencies.end(), 0);

	float visibilityEnergy = 0.0f;

	for (int i=0; i<numVoxels; i++)			//	only suitable for byte size datasets
	{
		int bin = (int)volume->memblock3D[i];

		visibilityEnergy -= colors[bin].a * histogram.visibilities[bin];
	}

	float constraintEnergy = 0.0f;
	float min = 0.0f;
	float max = 1.0f;

	for (int i=0; i<256; i++)
	{
		float opacity = colors[i].a;

		constraintEnergy += glm::max(glm::pow((min - opacity), 2.0f), 0.0f) + glm::max(glm::pow((opacity - max), 2.0f), 0.0f);
	}
}