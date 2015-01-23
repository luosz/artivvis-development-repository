#ifndef FREQUENCY_HISTOGRAM_H
#define FREQUENCY_HISTOGRAM_H

#include "Histogram.h"

class FrequencyHistogram  :  public Histogram
{
public:

	FrequencyHistogram()
	{
		numBins = 256;
		values.resize(numBins);
	}


	void Update(int currentTimestep, VolumeDataset &volume, GLuint tex3D, GLuint &tfTexture, ShaderManager &shaderManager, Camera &camera)
	{
		std::fill(values.begin(), values.end(), 0);

		for (int i=0; i<volume.numVoxels; i++)
		{
			int bucket = volume.memblock3D[(currentTimestep*volume.numVoxels) + i];
			values[bucket]++;
		}

		maxFrequency = 0;
		for (int i=1; i<256; i++)
		{
			int freq = values[i];
			maxFrequency = glm::max(maxFrequency, freq);
		}

		for (int i=1; i<256; i++)
		{
			values[i] /= (float)maxFrequency;
		}

		values[0] = 1.0f;
	}


private:
	int maxFrequency;
};

#endif