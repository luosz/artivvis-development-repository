#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <Windows.h>
#include <vector>
#include <algorithm>

using namespace std;

struct MyFileSort : public std::binary_function<std::string, std::string, bool>
{
	bool operator() (std::string &a, std::string &b) const
	{
		return (a.length() < b.length() || (a.length() == b.length() && (a < b)));
	}
};


int main(int argc, char** argv)
{
	string sourceFolder = "../../Samples/supernova/supernova/";
	string targetFileName = "../../Samples/supernova/DSSupernova/dsSupernova.";

	int xRes = 432;
	int yRes = 432;
	int zRes = 432;
	int numElements = xRes * yRes * zRes;
	int bytesPerElement = 4;

	int rawDataSize = numElements * bytesPerElement;
	char* buffer = new char[rawDataSize];

	std::vector<float> floats;
	floats.resize(numElements);

	std::vector<char> chars;
	chars.resize(numElements);

	WIN32_FIND_DATAA findFileData;
	HANDLE hFind;
	vector<string> files;

	float max = 0.0f;

	string search = sourceFolder + "*";

	hFind = FindFirstFileA(search.c_str(), &findFileData);

	if (hFind != INVALID_HANDLE_VALUE)
	{
		FindNextFileA(hFind, &findFileData);

		while (FindNextFileA(hFind, &findFileData) != 0)
			files.push_back(findFileData.cFileName);

		sort(files.begin(), files.end(), MyFileSort());

	}
	else
		cout << "Invalid Directory" << endl;


	cout << "Finding max float value" << endl;

	for (int i=0; i<files.size(); i++)
	{
		cout << i << ", ";

		int dataStartOffset;
		string line;
		streampos size;

		ifstream inStream (sourceFolder + files[i], std::ios::in|std::ios::binary|std::ios::ate);

		if (inStream.is_open())
		{
			size = inStream.tellg();
			inStream.seekg (0, std::ios::beg);

			inStream.read (buffer, rawDataSize);
			inStream.close();
		}
		else 
			std::cout << "Unable to open file";


		memcpy(&floats[0], buffer, rawDataSize);

		for (int j=0; j<numElements; j++)
		{
			if (floats[j] >  max)
				max = floats[j];
		}
	}


	std::cout << "Copying floats to chars.." << endl;

	// Very inefficient to do this all again but puts the least stress on memory and time constraints aren't really any issue

	for (int i=0; i<files.size(); i++)
	{
		cout << i << ", ";

		int dataStartOffset;
		string line;
		streampos size;

		ifstream inStream (sourceFolder + files[i], std::ios::in|std::ios::binary|std::ios::ate);

		if (inStream.is_open())
		{
			size = inStream.tellg();
			inStream.seekg (0, std::ios::beg);

			inStream.read (buffer, rawDataSize);
			inStream.close();
		}
		else 
			std::cout << "Unable to open file";




		memcpy(&floats[0], buffer, rawDataSize);
		float f;

		for (int j=0; j<numElements; j++)
		{
			f = floats[j] / max;
			chars[j] = f * 255;
		}
			



		ofstream outStream(targetFileName + to_string(i), std::ios::out|std::ios::binary);

		if (outStream.is_open())
		{
			outStream.write (&chars[0], numElements);
			outStream.close();
		}
		else 
			std::cout << "Unable to open file";
	}

	
	
    return 0;
}
