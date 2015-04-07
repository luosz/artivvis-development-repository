#include "VoxelReader.h"

// At the moment file is specified within function but in future will use parameter. 
void VoxelReader::LoadVolume(std::string folderPath, std::string headerFile, VolumeProperties &properties)
{
	// If folderPath or headerFile is empty, use predefined mhd header filename and folder path
	if (folderPath.length() <= 0 || headerFile.length() <= 0)
	{
		//	folderPath = "../../Samples/TVvort/";
		//	headerFile = folderPath + "TVvort.mhd";

		//	folderPath = "../../Samples/Nucleon/";
		//	headerFile = folderPath + "nucleon.mhd";

		//	folderPath = "../../Samples/CThead/";
		//	headerFile = folderPath + "CThead.mhd";

		//	folderPath = "../../Samples/MRbrain/";
		//	headerFile = folderPath + "MRbrain.mhd";

		//	folderPath = "../../Samples/TVlung/";
		//	headerFile = folderPath + "TVlung.mhd";

		//	folderPath = "../../Samples/FiveJets/";
		//	headerFile = folderPath + "FiveJetsDensity.mhd";

		//	folderPath = "../../Samples/Isabel/";
		//	headerFile = folderPath + "IsabelCloud.mhd";

		//	folderPath = "../../Samples/TJet/";
		//	headerFile = folderPath + "TJet.mhd";

		//	folderPath = "../../Samples/Abdomen/";
		//	headerFile = folderPath + "Abdomen16.mhd";

		//	folderPath = "../../Samples/Colon/";
		//	headerFile = folderPath + "colon.mhd";

		//	folderPath = "../../Samples/MRThead/";
		//	headerFile = folderPath + "MRThead.mhd";

		//	folderPath = "../../Samples/MRIhead/";
		//	headerFile = folderPath + "MRIhead.mhd";

		//	folderPath = "../../Samples/Bonsai/";
		//	headerFile = folderPath + "bonsai.mhd";

		//	folderPath = "../../Samples/SmokeSim/";
		//	headerFile = folderPath + "SmokeSim.mhd";

		//	folderPath = "../../Samples/SmokeSim/";
		//	headerFile = folderPath + "SmokeSimBig.mhd";

			folderPath = "../../Samples/SmokeSim/";
			headerFile = folderPath + "SmokeSimSideways.mhd";

		//	folderPath = "../../Samples/CTknee/";
		//	headerFile = folderPath + "CTknee.mhd";

		//	folderPath = "../../Samples/downsampled vortex/";
		//	headerFile = folderPath + "dsVort.mhd";
	}

	ReadMHD(folderPath, headerFile, properties);
	ReadRaw(properties);
}


// Reads a header of .mhd format and copies values to 'VolumeProperties'
void VoxelReader::ReadMHD(std::string folderPath, std::string headerFile, VolumeProperties &properties)
{
	std::ifstream myFile(headerFile.c_str());

	if (!myFile.is_open())
	{
		std::cout << "[VoxelReader] Failed to open file: " << headerFile.c_str() << std::endl;
		return;
	}

	std::vector<std::string> fileLines;
	std::string line, temp, temp2;
	std::istringstream iss;

	while (getline(myFile, line))
		fileLines.push_back(line);

	for (int i = 0; i < fileLines.size(); i++)
	{
		iss.str(fileLines[i]);
		iss >> temp;
		iss >> temp2;

		if (temp == "Timesteps")
		{
			iss >> properties.timesteps;
		}
		else if (temp == "NDims")
		{
			iss >> properties.numDims;
		}
		else if (temp == "DimSize")
		{
			iss >> properties.xRes;
			iss >> properties.yRes;
			iss >> properties.zRes;
		}
		else if (temp == "ElementType")
		{
			iss >> properties.elementType;

			if (properties.elementType == "MET_UCHAR")
				properties.bytesPerElement = 1;
			else if (properties.elementType == "SHORT")
				properties.bytesPerElement = 2;
			else if (properties.elementType == "FLOAT")
				properties.bytesPerElement = 4;
		}
		else if (temp == "ElementDataFile")
		{
			iss >> temp;
			properties.rawFilePath = folderPath;
			properties.rawFilePath.append(temp);
		}
		else if (temp == "ElementByteOrderMSB")
		{
			iss >> temp;

			if (temp == "TRUE" || temp == "True")
				properties.littleEndian = true;
			else
				properties.littleEndian = false;
		}
		iss.clear();
	}
}


// Basic sort function used to sort files numerically rather than per character
struct MyFileSort : public std::binary_function<std::string, std::string, bool>
{
	bool operator() (std::string &a, std::string &b) const
	{
		return (a.length() < b.length() || (a.length() == b.length() && (a < b)));
	}
};

bool NumericalFileSort(const std::string a, const std::string b)
{
	return (a.length() < b.length() || (a.length() == b.length() && (a < b)));
};


// Reads in the raw binary data using properties copied in from header
void VoxelReader::ReadRaw(VolumeProperties &properties)
{
	//	int bufferSize = properties.xRes * properties.yRes * properties.zRes * properties.bytesPerElement * properties.timesteps;
	int bufferSize = properties.xRes * properties.yRes * properties.zRes * properties.bytesPerElement;
	properties.bufferAddress = new GLubyte[bufferSize];
	int numBytesInBufferFilled = 0;

	std::string directory = properties.rawFilePath;
//	directory.append("/*");


	struct stat status;
	stat(directory.c_str(), &status);
	if (status.st_mode & S_IFDIR)
	{
		tinydir_dir dir;
		tinydir_open(&dir, directory.c_str());

		while (dir.has_next)
		{
			tinydir_file file;
			tinydir_readfile(&dir, &file);

			if (file.is_reg)
				files.push_back(std::string(file.name));

			tinydir_next(&dir);
		}

		sort(files.begin(), files.end(), NumericalFileSort);

		for (int i = 0; i < files.size(); i++)
			files[i] = std::string(properties.rawFilePath + "/" + files[i]);


		CopyFileToBuffer(files[0], numBytesInBufferFilled, properties);
	}
	else
	{
		CopyFileToBuffer(properties.rawFilePath, numBytesInBufferFilled, properties);
	}
}


// Current file to be copied to buffer. Offset is for multiple files being read into single buffer, set as file offset
void VoxelReader::CopyFileToBuffer(std::string fileName, int &numBytesInBufferFilled, VolumeProperties &properties)
{
	std::streampos size;

	std::ifstream myFile(fileName.c_str(), std::ios::in | std::ios::binary | std::ios::ate);

	if (myFile.is_open())
	{
		size = myFile.tellg();

		myFile.seekg(0, std::ios::beg);
		myFile.read((char*)properties.bufferAddress + numBytesInBufferFilled, size);
		myFile.close();

		numBytesInBufferFilled += size;
	}
	else
		std::cout << "Unable to open file: " << fileName.c_str() << std::endl;


}


void VoxelReader::CopyFileToBuffer(GLubyte* bufferAddress, int fileIndex)
{
	std::string fileName = files[fileIndex];

	std::streampos size;

	std::ifstream myFile(fileName.c_str(), std::ios::in | std::ios::binary | std::ios::ate);

	if (myFile.is_open())
	{
		size = myFile.tellg();

		myFile.seekg(0, std::ios::beg);
		myFile.read((char*)bufferAddress, size);
		myFile.close();
	}
	else
		std::cout << "Unable to open file" << std::endl;


}


// Initializes the values found in VolumeProperties which might not be specified in the header
VolumeProperties::VolumeProperties()
{
	timesteps = 1;
	timePerFrame = 0.1f;
}

