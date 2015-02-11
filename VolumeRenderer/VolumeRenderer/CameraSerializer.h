#ifndef CameraSerializer_h
#define CameraSerializer_h
#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "Camera.h"

class CameraSerializer
{
public:
	static void to_file(const Camera &camera, const char *filename)
	{
		std::stringstream ss;
		ss << camera.position.x << "\t" << camera.position.y << "\t" << camera.position.z << std::endl;
		ss << camera.xPixels << "\t" << camera.yPixels << std::endl;
		std::ofstream fs(filename);
		if (fs.is_open())
		{
			fs << ss.str();
			fs.close();
		}
		else
		{
			std::cout << "Error opening file " << filename << std::endl;
		}
	}

	static Camera from_file(const char *filename)
	{
		std::ifstream fs(filename);
		if (fs.is_open())
		{
			glm::vec3 position;
			int xPixels, yPixels;
			fs >> position.x >> position.y >> position.z;
			fs >> xPixels >> yPixels;
			fs.close();
			Camera camera(xPixels, yPixels, position);
			return camera;
		}
		else
		{
			std::cout << "Error opening file " << filename << std::endl;
			return Camera();
		}
	}
};

#endif // CameraSerializer_h
