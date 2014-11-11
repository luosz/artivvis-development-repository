#include "TransferFunction.h"

void TransferFunction::Init(const char *filename, VolumeDataset &volume_)
{
	colorTable.resize(256);
	LoadXML(filename);
	LoadLookup();

	volume = &volume_;
	frequencies.resize(256);
	CalculateFrequencies();
	weights.resize(numIntensities);
	numIterations = 1000;

	targetIntensity = 0.5f;
}

void TransferFunction::LoadXML(const char *filename)
{
	tinyxml2::XMLDocument doc;
//	auto r = doc.LoadFile("nucleon.tfi");
	auto r = doc.LoadFile("CT-Knee_spectrum_16_balance.tfi");
//	auto r = doc.LoadFile("../../Samples/downsampled vortex/90.tfi");

	if (r != tinyxml2::XML_NO_ERROR)
		std::cout << "failed to open file" << std::endl;

	auto transFuncIntensity = doc.FirstChildElement("VoreenData")->FirstChildElement("TransFuncIntensity");

	auto key = doc.FirstChildElement("VoreenData")->FirstChildElement("TransFuncIntensity")->FirstChildElement("Keys")->FirstChildElement("key");

	while (key)
	{
		float intensity = atof(key->FirstChildElement("intensity")->Attribute("value"));
		intensities.push_back(intensity);

		int r = atoi(key->FirstChildElement("colorL")->Attribute("r"));
		int g = atoi(key->FirstChildElement("colorL")->Attribute("g"));
		int b = atoi(key->FirstChildElement("colorL")->Attribute("b"));
		int a = atoi(key->FirstChildElement("colorL")->Attribute("a"));

		colors.push_back(glm::vec4(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f));

		std::cout << "intensity=" << intensity;
		std::cout << "\tcolorL r=" << r << " g=" << g << " b=" << b << " a=" << a;
		std::cout << std::endl;

		key = key->NextSiblingElement();
	}

	numIntensities = intensities.size();
}

void TransferFunction::LoadLookup()
{
	glm::vec4 previousColor(0.0f);
	float previousIntensity = 0.0f;
	int next = 0;

	for (int i=0; i<256; i++)
	{
		float currentIntensity = (float)i / (float)256;

		while (next < numIntensities && currentIntensity > intensities[next])
		{
			previousIntensity = intensities[next];
			previousColor = colors[next];
			next++;
		}

		if (next < numIntensities)
			colorTable[i] = LERPColor(previousColor, colors[next], previousIntensity, intensities[next], currentIntensity);
		else
			colorTable[i] = LERPColor(previousColor, glm::vec4(0.0f), previousIntensity, 1.0f, currentIntensity);
	}

	glGenTextures(1, &tfTexture);
    glBindTexture(GL_TEXTURE_1D, tfTexture);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 256, 0, GL_RGBA, GL_FLOAT, &colorTable[0]); 
	glBindTexture(GL_TEXTURE_1D, 0);
}

glm::vec4 TransferFunction::LERPColor(glm::vec4 firstColor, glm::vec4 secondColor, float firstIntensity, float secondIntensity, float currentIntensity)
{
	float fraction = (currentIntensity - firstIntensity) / (secondIntensity - firstIntensity);

	return firstColor + ((secondColor - firstColor) * fraction);
}












void TransferFunction::CalculateFrequencies()
{
	std::fill(frequencies.begin(), frequencies.end(), 0);

	numVoxels = volume->xRes * volume->yRes * volume->zRes;

	for (int i=0; i<numVoxels; i++)			//	only suitable for byte size datasets
	{
		int bin = (int)volume->memblock3D[i];
		frequencies[bin]++;
	}
}

float TransferFunction::GetWeightedEntropyOpacityByID(float intensity, int index)
{
	float frequency = frequencies[intensity];
	//const double epsilon = 1e-6;
	float probability = frequency / numVoxels;

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

float TransferFunction::GetWeightedAreaEntropy(int index)
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

	float sum = 0;
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


void TransferFunction::IntensityOptimize()
{
	int sum = 0;

	for (int i=0; i<numIntensities; i++)
	{
		float dist = glm::abs(intensities[i] - targetIntensity);
		weights[i] = dist;
		sum += dist;
	}

	if (sum > 0)
	{
		for (int i=0; i<numIntensities; i++)
			weights[i] /= sum;
	}
	
	for (int i=0; i<numIterations; i++)
	{
		int max_index = -1;
		int min_index = -1;
		float max_area = std::numeric_limits<float>::min();
		float min_area = std::numeric_limits<float>::max();

		for (int j=0; j<numIntensities - 1; j++)
		{
			if (colors[j].a > glm::epsilon<float>())
			{
				float area = GetWeightedAreaEntropy(j);
				if (area > max_area)
				{
					max_index = j;
					max_area = area;
				}
				if (area < min_area && colors[j].a < 1.0f)
				{
					min_index = j;
					min_area = area;
				}
			}
		}


		if (min_index != max_index && max_index > -1 && min_index > -1)
		{
			// get the upper vertex of an edge
			int max_index_next = max_index + 1;

			float weight_max_1 = GetWeightedEntropyOpacityByID(intensities[max_index] * 255.0f, max_index);
			float weight_max_2 = GetWeightedEntropyOpacityByID(intensities[max_index_next] * 255.0f, max_index_next);

			if (colors[max_index_next].a > glm::epsilon<float>() && colors[max_index_next].a < 1 && weight_max_2 > weight_max_1)
			{
				max_index++;
			}

			// get the lower vertex of an edge
			int min_index_next = min_index + 1;

			float weight_min_1 = GetWeightedEntropyOpacityByID(intensities[min_index] * 255.0f, min_index);
			float weight_min_2 = GetWeightedEntropyOpacityByID(intensities[min_index_next] * 255.0f, min_index_next);

			if (colors[min_index_next].a > glm::epsilon<float>() && colors[min_index_next].a < 1 && weight_min_2 < weight_min_1)
			{
				min_index++;
			}

			float step_size = 1.0f / 255.0f;

			float height_max = colors[max_index].a;
			float height_max_new = height_max - step_size;
			height_max_new = height_max_new < glm::epsilon<float>() ? glm::epsilon<float>() : height_max_new;

			float area = GetWeightedNeighbourArea(max_index);
			colors[max_index].a = height_max_new; // update opacity
			float new_area = GetWeightedNeighbourArea(max_index); // calculate new area using new opacity
			float area_decreased = area - new_area;

			//double height_increased = get_height_given_area_increment(min_index, area_decreased);
			float height_min = colors[min_index].a;
			float height_min_new = height_min + step_size;
			height_min_new = height_min_new > 1 ? 1 : height_min_new;
			colors[min_index].a = height_min_new; // update opacity
			//std::cout<<"balance TF entropy max index="<<max_index<<" min index="<<min_index<<" opacity="<<height_max<<" new opacity="<<height_max_new<<" area="<<area<<" new area="<<new_area<<" height="<<height_min<<" new height="<<height_min_new<<endl;
		}
	}

	LoadLookup();
}

float TransferFunction::GetWeightedNeighbourArea(int index)
{
	return GetWeightedAreaEntropy(index) + GetWeightedAreaEntropy(index - 1);
}


float TransferFunction::GetOpacityByInterp(float intensity, int index)
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


float TransferFunction::GetWeightByInterp(float intensity, int index)
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

float TransferFunction::GetWeight(int index)
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