#include "IntensityTFOptimizerV2.h"

IntensityTFOptimizerV2::IntensityTFOptimizerV2(VolumeDataset *volume_, TransferFunction *transferFunction_, VisibilityHistogram *visibilityHistogram_)
{
	volume = volume_;
	transferFunction = transferFunction_;
	visibilityHistogram = visibilityHistogram_;

	frequencies.resize(256);
	CalculateFrequencies();
	weights.resize(transferFunction->numIntensities);
	numIterations = 1000;	
}

void IntensityTFOptimizerV2::CalculateFrequencies()
{
	std::fill(frequencies.begin(), frequencies.end(), 0);

	numVoxels = volume->xRes * volume->yRes * volume->zRes;

	for (int i=0; i<numVoxels; i++)			//	only suitable for byte size datasets
	{
		int bin = (int)volume->memblock3D[i];
		frequencies[bin]++;
	}
}

float IntensityTFOptimizerV2::GetEntropyOpacityByID(float intensity, int index)
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


float IntensityTFOptimizerV2::GetVisibilityOpacityByID(float intensity, int index)
{
	int bin_index = static_cast<int>(intensity);
	float frequency = frequencies[bin_index];
	float probability = frequency / (float)numVoxels;

	if (probability > glm::epsilon<float>())
	{
		if (visibilityHistogram)
		{
			float visibility = visibilityHistogram->visibilities[bin_index];
			float normalised = intensity / 255.0f;
			return GetOpacityByInterp(normalised, index) * visibility * frequency;
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


float IntensityTFOptimizerV2::GetAreaEntropy(int index)
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


float IntensityTFOptimizerV2::GetAreaVisibility(int index)
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

void IntensityTFOptimizerV2::Optimize()
{
	OptimizeForIntensity();
}

void IntensityTFOptimizerV2::OptimizeForIntensity()
{
	BeginOfOptimization();
	//////////////////////////////////////////////////////////////////////////

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
			float area = GetWeightedAreaEntropy(i);
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
					int max_index = i;
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
					int min_index = i;
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

	//////////////////////////////////////////////////////////////////////////
	EndOfOptimization();
}

void IntensityTFOptimizerV2::BalanceEdges()
{
	BeginOfOptimization();
	//////////////////////////////////////////////////////////////////////////

	for (int i = 0; i < numIterations; i++)
	{
		const float step_size = 1.0f / 255.0f;
		std::vector<double> area_list;
		double mean_area = 0;

		for (int i = 0; i < numIntensities - 1; i++)
		{
			float area = GetAreaEntropy(i);
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
					int max_index = i;
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
					int min_index = i;
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

	//////////////////////////////////////////////////////////////////////////
	EndOfOptimization();
}

void IntensityTFOptimizerV2::BalanceVisibilityOnce()
{
	BeginOfOptimizationNoCopy();
	//////////////////////////////////////////////////////////////////////////

	const float step_size = 1.0f / 255.0f;
	std::vector<double> area_list;
	double mean_area = 0;

	for (int i = 0; i < numIntensities - 1; i++)
	{
		float area = GetAreaVisibility(i);
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
				int max_index = i;
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
				int min_index = i;
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

	//////////////////////////////////////////////////////////////////////////
	EndOfOptimization();
}

void IntensityTFOptimizerV2::BalanceVisibility()
{
	BeginOfOptimization();
	//////////////////////////////////////////////////////////////////////////

	for (int i = 0; i < numIterations; i++)
	{
		const float step_size = 1.0f / 255.0f;
		std::vector<double> area_list;
		double mean_area = 0;

		for (int i = 0; i < numIntensities - 1; i++)
		{
			float area = GetAreaVisibility(i);
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
					int max_index = i;
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
					int min_index = i;
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

	//////////////////////////////////////////////////////////////////////////
	EndOfOptimization();
}

float IntensityTFOptimizerV2::GetWeightedEntropyOpacityByID(float intensity, int index)
{
	int bin_index = static_cast<int>(intensity);
	float frequency = frequencies[bin_index];
	float probability = frequency / (float)numVoxels;

	if (probability > glm::epsilon<float>())
	{
		float normalised = intensity / 255.0f;
		return GetWeightByInterp(normalised, index) * GetOpacityByInterp(normalised, index) * probability * (-log(probability));
	}
	else
	{
		return 0;
	}
}

float IntensityTFOptimizerV2::GetOpacityByInterp(float intensity, int index)
{
	int i1 = index;
	int i2 = index + 1;

	if (i1 >= 0 && i2 < numIntensities)
	{
		// linear interpolation
		float t = (intensity - intensities[i1]) / (intensities[i2] - intensities[i1]);

		float a = colors[i1].a;
		float b = colors[i2].a;

		return (a + (b - a) * t);
	}
	else
	{
		if (i1 == -1)
		{
			return colors[i2].a;
		}
		else
		{
			if (i1 == numIntensities - 1)
			{
				return colors[i1].a;
			}
			else
			{
				std::cout << "Errors occur in get_opacity_by_interpolation()" << std::endl;
				return 0;
			}
		}
	}
}

float IntensityTFOptimizerV2::GetWeightByInterp(float intensity, int index)
{
	int i1 = index;
	int i2 = index + 1;

	if (i1 >= 0 && i2 < numIntensities)
	{
		// linear interpolation
		float t = (intensity - intensities[i1]) / (intensities[i2] - intensities[i1]);

		// get control point weights
		float a = GetWeight(i1);
		float b = GetWeight(i2);

		return (a + (b - a) * t);
	}
	else
	{
		if (i1 == -1)
		{
			return GetWeight(i2);
		}
		else
		{
			if (i1 == numIntensities - 1)
			{
				return GetWeight(i1);
			}
			else
			{
				std::cout << "Errors occur in get_control_point_weight_by_interpolation()" << std::endl;
				return 0;
			}
		}
	}
}

float IntensityTFOptimizerV2::GetWeight(int index)
{
	if (index >= 0 && index < numIntensities)
	{
		return weights[index];
	}
	else
	{
		return 1;
	}
}

float IntensityTFOptimizerV2::GetWeightedAreaEntropy(int index)
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
			sum += GetWeightedEntropyOpacityByID(intensity, index);
		}
	}
	//std::cout<<std::endl;
	return sum;
}
