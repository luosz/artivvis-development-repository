#ifndef FREQUENCY_HISTOGRAM_H
#define FREQUENCY_HISTOGRAM_H

#include "Histogram.h"
#include <thread>
#include <atomic>

class FrequencyHistogram  :  public Histogram
{
public:
	std::atomic<int> atomicValues[256];
	std::vector<std::thread> threads;
	int numThreads;

	FrequencyHistogram()
	{
		numBins = 256;
		values.resize(numBins);

		numThreads = 7;
		threads.resize(numThreads);
	}


	void Bin(int begin, int end, int currentTimestep, VolumeDataset &volume)
	{
		for (int i=begin; i<end; i++)
		{
			int bucket = volume.memblock3D[(currentTimestep*volume.numVoxels) + i];

			if (bucket != 0)
				atomicValues[bucket].fetch_add((int)1);
		}
	}

	void Update(int currentTimestep, VolumeDataset &volume, GLuint tex3D, GLuint &tfTexture, ShaderManager &shaderManager, Camera &camera)
	{
		for (int i=0; i<numBins; i++)
			atomicValues[i].store(0);
		
		int beginID = 0;
		int numPerThread = volume.numVoxels / numThreads;

		for (int i=0; i<numThreads-1; i++)
		{
			threads[i] = std::thread(&FrequencyHistogram::Bin, this, beginID, beginID + numPerThread, currentTimestep, std::ref(volume));
			beginID += numPerThread;
		}
		threads[numThreads-1] = std::thread(&FrequencyHistogram::Bin, this, beginID, volume.numVoxels, currentTimestep, std::ref(volume));

		for (int i=0; i<numThreads; i++)
			threads[i].join();

		maxFrequency = 0;
		for (int i=1; i<256; i++)
		{
			int freq = atomicValues[i];
			maxFrequency = glm::max(maxFrequency, freq);
		}

		for (int i=1; i<256; i++)
		{
			values[i] = atomicValues[i] / (float)maxFrequency;
		}

		values[0] = 1.0f;
	}

	
//	void Update(int currentTimestep, VolumeDataset &volume, GLuint tex3D, GLuint &tfTexture, ShaderManager &shaderManager, Camera &camera)
//	{
//		std::fill(values.begin(), values.end(), 0);
//
//		for (int i=0; i<volume.numVoxels; i++)
//		{
//			int bucket = volume.memblock3D[(currentTimestep*volume.numVoxels) + i];
//			values[bucket]++;
//		}
//
//		maxFrequency = 0;
//		for (int i=1; i<256; i++)
//		{
//			int freq = values[i];
//			maxFrequency = glm::max(maxFrequency, freq);
//		}
//
//		for (int i=1; i<256; i++)
//		{
//			values[i] /= (float)maxFrequency;
//		}
//
//		values[0] = 1.0f;
//	}
	

private:
	int maxFrequency;
};

#endif