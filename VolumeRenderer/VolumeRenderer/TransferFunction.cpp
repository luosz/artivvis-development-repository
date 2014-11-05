#include "TransferFunction.h"

void TransferFunction::Init(const char *filename, VolumeDataset &volume_)
{
	colorTable.resize(256);
	LoadXML(filename);
	LoadLookup();

	volume = &volume_;
	CalculateFrequencies();
	weights.resize(numIntensities);
}

void TransferFunction::CalculateFrequencies()
{
	frequencies.resize(256);
	std::fill(frequencies.begin(), frequencies.end(), 0);

	numVoxels = volume->xRes * volume->yRes * volume->zRes;

	for (int i=0; i<numVoxels; i++)
	{
		float scalar = (float)volume->memblock3D[i];
		int bin = (int)(scalar * 256.0f);
		frequencies[bin]++;
	}
}


void TransferFunction::LoadXML(const char *filename)
{
	tinyxml2::XMLDocument doc;
	auto r = doc.LoadFile("nucleon.tfi");
//	auto r = doc.LoadFile("../../Samples/CTknee/transfer_function/CT-Knee_spectrum_16_balance.tfi");
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



float TransferFunction::GetEntropyByID(float intensity, int index)
{
	float frequency = frequencies[(int)(intensity * 256)];
		//const double epsilon = 1e-6;
	float probability = frequency / numVoxels;

	if (probability > glm::epsilon<float>())
	{
		float normalised = intensity / 256.0f;

		return GetWeightByInterp(normalised, index) * GetOpacityByInterp(normalised, index) * probability * (-log(probability));
	}
	else
	{
		return 0;
	}
}

float TransferFunction::GetEntropy(int index)
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
				std::cout << "index out of range in get_area_integral()" << std::endl;
			}
		}
	}

	//std::cout<<"intensity "<<a<<" "<<b;
	a = a * 256.0f;
	b = b * 256.0f;
	//std::cout<<" map to [0, 255] "<<a<<" "<<b<<std::endl;

	float sum = 0;
	// int intensity belongs to [0,255]
	for (int intensity = (int)a; intensity < b; intensity++)
	{
		if (intensity >= a)
		{
			//std::cout<<intensity<<" ";
			sum += GetEntropyByID(intensity, index);
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
	

	int max_index = -1;
	int min_index = -1;
	float max_area = std::numeric_limits<float>::min();
	float min_area = std::numeric_limits<float>::max();


	for (int i = 0; i<numIntensities - 1; i++)
	{
		if (colors[i].a > glm::epsilon<float>())
		{
			float area = GetEntropy(i);
			if (area > max_area)
			{
				max_index = i;
				max_area = area;
			}
			if (area < min_area && colors[i].a < 1.0f)
			{
				min_index = i;
				min_area = area;
			}
		}
	}
}


float TransferFunction::GetOpacityByInterp(float intensity, int index)
{
	int i1 = index, i2 = index + 1;
	if (i1 >= 0 && i2 < numIntensities)
	{
		// linear interpolation
		double t = (intensity - intensities[i1]) / (intensities[i2] - intensities[i1]);

		double a = intensities[i1];
		double b = intensities[i2];
		return (a + (b - a) * t);
	}
	else
	{
		if (i1 == -1)
		{
			return intensities[i2];
		}
		else
		{
			if (i1 == numIntensities - 1)
			{
				return intensities[i1];
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
	int i1 = index, i2 = index + 1;

	if (i1 >= 0 && i2 < numIntensities)
	{
		// linear interpolation
		double t = (intensity - intensities[i1]) / (intensities[i2] - intensities[i1]);

		// get control point weights
		double a = GetWeight(i1);
		double b = GetWeight(i2);

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