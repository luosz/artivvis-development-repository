#pragma once

#ifndef IntensityTFOptimizerV2_h
#define IntensityTFOptimizerV2_h

#include "VolumeDataset.h"
#include "TransferFunction.h"
#include "VisibilityHistogram.h"
#include "IntensityTFOptimizer.h"

class IntensityTFOptimizerV2 : public IntensityTFOptimizer
{
public:
	VisibilityHistogram *visibilityHistogram;

	IntensityTFOptimizerV2(VolumeDataset &volume_, int numIntensities_, glm::vec4 *colors_, float *intensities_) : IntensityTFOptimizer(volume_, numIntensities_, colors_, intensities_)
	{
		visibilityHistogram = NULL;
	}

	float GetEntropyOpacityByID(float intensity, int index)
	{
		int bin_index = static_cast<int>(intensity);
		float frequency = frequencies[bin_index];
		float probability = frequency / (float)numVoxels;

		if (probability > glm::epsilon<float>())
		{
			float normalised = intensity / 255.0f;
			return GetOpacityByInterp(normalised, index) * probability * (-log(probability));
		}
		else
		{
			return 0;
		}
	}

	float GetVisibilityOpacityByID(float intensity, int index)
	{
		int bin_index = static_cast<int>(intensity);
		float frequency = frequencies[bin_index];
		float probability = frequency / (float)numVoxels;

		if (probability > glm::epsilon<float>())
		{
			if (visibilityHistogram)
			{
				auto visibility = visibilityHistogram->visibilities[bin_index];
				float normalised = intensity / 255.0f;
				return GetOpacityByInterp(normalised, index) * visibility;
			} 
			else
			{
				std::cout << "Error: visibilityHistogram is NULL" << std::endl;
				return 0;
			}
		}
		else
		{
			return 0;
		}
	}

	float GetAreaEntropy(int index)
	{
		float a, b;

		if (index >= 0 && index < numIntensities - 1)
		{
			a = intensities[index];
			b = intensities[index + 1];
		}
		else
		{
			if (index == -1)
			{
				a = 0.0f;
				b = intensities[index + 1];
			}
			else
			{
				if (index == numIntensities - 1)
				{
					a = intensities[index];
					b = 1.0f;
				}
				else
				{
					std::cout << "index out of range in get_area_entropy()" << std::endl;
				}
			}
		}

		//std::cout<<"intensity "<<a<<" "<<b;
		a = a * 255.0f;
		b = b * 255.0f;
		//std::cout<<" map to [0, 255] "<<a<<" "<<b<<std::endl;

		float sum = 0.0f;
		// int intensity belongs to [0,255]
		for (int intensity = (int)a; intensity < b; intensity++)
		{
			if (intensity >= a)
			{
				//std::cout<<intensity<<" ";
				sum += GetEntropyOpacityByID(intensity, index);
			}
		}
		//std::cout<<std::endl;
		return sum;
	}

	float GetAreaVisibility(int index)
	{
		float a, b;

		if (index >= 0 && index < numIntensities - 1)
		{
			a = intensities[index];
			b = intensities[index + 1];
		}
		else
		{
			if (index == -1)
			{
				a = 0.0f;
				b = intensities[index + 1];
			}
			else
			{
				if (index == numIntensities - 1)
				{
					a = intensities[index];
					b = 1.0f;
				}
				else
				{
					std::cout << "index out of range in get_area_entropy()" << std::endl;
				}
			}
		}

		//std::cout<<"intensity "<<a<<" "<<b;
		a = a * 255.0f;
		b = b * 255.0f;
		//std::cout<<" map to [0, 255] "<<a<<" "<<b<<std::endl;

		float sum = 0.0f;
		// int intensity belongs to [0,255]
		for (int intensity = (int)a; intensity < b; intensity++)
		{
			if (intensity >= a)
			{
				//std::cout<<intensity<<" ";
				sum += GetVisibilityOpacityByID(intensity, index);
			}
		}
		//std::cout<<std::endl;
		return sum;
	}

	virtual void Optimize(float targetIntensity)
	{
		float sum = 0.0f;

		for (int i = 0; i<numIntensities; i++)
		{
			float dist = glm::abs(intensities[i] - targetIntensity);
			weights[i] = dist;
			sum += dist;
		}

		if (sum > 0.0f)
		{
			for (int i = 0; i < numIntensities; i++)
				weights[i] /= sum;
		}

		for (int i = 0; i < numIterations; i++)
		{
			const float step_size = 1.0f / 255.0f;
			std::vector<double> area_list;
			double mean_area = 0;

			for (int i = 0; i < numIntensities - 1; i++)
			{
				auto area = GetWeightedAreaEntropy(i);
				area_list.push_back(area);
				mean_area += area;
			}
			mean_area = mean_area / area_list.size();

			for (int i = 0; i < area_list.size(); i++)
			{
				// move only non-zero control points
				if (colors[i].a > glm::epsilon<float>())
				{
					if (area_list[i] > mean_area)
					{
						// get the upper vertex of an edge
						auto max_index = i;
						int max_index_next = max_index + 1;
						float weight_max_1 = GetWeightedEntropyOpacityByID(intensities[max_index] * 255.0f, max_index);
						float weight_max_2 = GetWeightedEntropyOpacityByID(intensities[max_index_next] * 255.0f, max_index_next);
						if (colors[max_index_next].a > glm::epsilon<float>() && colors[max_index_next].a < 1.0f && weight_max_2 > weight_max_1)
						{
							max_index++;
						}

						float height_max = colors[max_index].a;
						float height_max_new = height_max - step_size;
						height_max_new = height_max_new < glm::epsilon<float>() ? glm::epsilon<float>() : height_max_new;
						colors[max_index].a = height_max_new; // update opacity
					}
					if (area_list[i] < mean_area)
					{
						auto min_index = i;
						// get the lower vertex of an edge
						int min_index_next = min_index + 1;
						float weight_min_1 = GetWeightedEntropyOpacityByID(intensities[min_index] * 255.0f, min_index);
						float weight_min_2 = GetWeightedEntropyOpacityByID(intensities[min_index_next] * 255.0f, min_index_next);
						if (colors[min_index_next].a > glm::epsilon<float>() && colors[min_index_next].a < 1.0f && weight_min_2 < weight_min_1)
						{
							min_index++;
						}

						float height_min = colors[min_index].a;
						float height_min_new = height_min + step_size;
						height_min_new = height_min_new > 1.0f ? 1.0f : height_min_new;
						colors[min_index].a = height_min_new; // update opacity
					}
				}
			}
		}
	}

	void BalanceEdges()
	{
		for (int i = 0; i < numIterations; i++)
		{
			const float step_size = 1.0f / 255.0f;
			std::vector<double> area_list;
			double mean_area = 0;

			for (int i = 0; i < numIntensities - 1; i++)
			{
				auto area = GetAreaEntropy(i);
				area_list.push_back(area);
				mean_area += area;
			}
			mean_area = mean_area / area_list.size();

			for (int i = 0; i < area_list.size(); i++)
			{
				// move only non-zero control points
				if (colors[i].a > glm::epsilon<float>())
				{
					if (area_list[i] > mean_area)
					{
						// get the upper vertex of an edge
						auto max_index = i;
						int max_index_next = max_index + 1;
						float weight_max_1 = GetEntropyOpacityByID(intensities[max_index] * 255.0f, max_index);
						float weight_max_2 = GetEntropyOpacityByID(intensities[max_index_next] * 255.0f, max_index_next);
						if (colors[max_index_next].a > glm::epsilon<float>() && colors[max_index_next].a < 1.0f && weight_max_2 > weight_max_1)
						{
							max_index++;
						}

						float height_max = colors[max_index].a;
						float height_max_new = height_max - step_size;
						height_max_new = height_max_new < glm::epsilon<float>() ? glm::epsilon<float>() : height_max_new;
						colors[max_index].a = height_max_new; // update opacity
					}
					if (area_list[i] < mean_area)
					{
						auto min_index = i;
						// get the lower vertex of an edge
						int min_index_next = min_index + 1;
						float weight_min_1 = GetEntropyOpacityByID(intensities[min_index] * 255.0f, min_index);
						float weight_min_2 = GetEntropyOpacityByID(intensities[min_index_next] * 255.0f, min_index_next);
						if (colors[min_index_next].a > glm::epsilon<float>() && colors[min_index_next].a < 1.0f && weight_min_2 < weight_min_1)
						{
							min_index++;
						}

						float height_min = colors[min_index].a;
						float height_min_new = height_min + step_size;
						height_min_new = height_min_new > 1.0f ? 1.0f : height_min_new;
						colors[min_index].a = height_min_new; // update opacity
					}
				}
			}
		}
	}

	void SetVisibilityHistogram(VisibilityHistogram &visibilityHistogram)
	{
		this->visibilityHistogram = &visibilityHistogram;
	}

	void BalanceVisibility()
	{
		for (int i = 0; i < numIterations; i++)
		{
			const float step_size = 1.0f / 255.0f;
			std::vector<double> area_list;
			double mean_area = 0;

			for (int i = 0; i < numIntensities - 1; i++)
			{
				auto area = GetAreaVisibility(i);
				area_list.push_back(area);
				mean_area += area;
			}
			mean_area = mean_area / area_list.size();

			for (int i = 0; i < area_list.size(); i++)
			{
				// move only non-zero control points
				if (colors[i].a > glm::epsilon<float>())
				{
					if (area_list[i] > mean_area)
					{
						// get the upper vertex of an edge
						auto max_index = i;
						int max_index_next = max_index + 1;
						float weight_max_1 = GetVisibilityOpacityByID(intensities[max_index] * 255.0f, max_index);
						float weight_max_2 = GetVisibilityOpacityByID(intensities[max_index_next] * 255.0f, max_index_next);
						if (colors[max_index_next].a > glm::epsilon<float>() && colors[max_index_next].a < 1.0f && weight_max_2 > weight_max_1)
						{
							max_index++;
						}

						float height_max = colors[max_index].a;
						float height_max_new = height_max - step_size;
						height_max_new = height_max_new < glm::epsilon<float>() ? glm::epsilon<float>() : height_max_new;
						colors[max_index].a = height_max_new; // update opacity
					}
					if (area_list[i] < mean_area)
					{
						auto min_index = i;
						// get the lower vertex of an edge
						int min_index_next = min_index + 1;
						float weight_min_1 = GetVisibilityOpacityByID(intensities[min_index] * 255.0f, min_index);
						float weight_min_2 = GetVisibilityOpacityByID(intensities[min_index_next] * 255.0f, min_index_next);
						if (colors[min_index_next].a > glm::epsilon<float>() && colors[min_index_next].a < 1.0f && weight_min_2 < weight_min_1)
						{
							min_index++;
						}

						float height_min = colors[min_index].a;
						float height_min_new = height_min + step_size;
						height_min_new = height_min_new > 1.0f ? 1.0f : height_min_new;
						colors[min_index].a = height_min_new; // update opacity
					}
				}
			}
		}
	}
};

#endif // IntensityTFOptimizerV2_h
