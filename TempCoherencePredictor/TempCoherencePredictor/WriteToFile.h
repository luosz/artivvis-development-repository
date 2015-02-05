#ifndef WRITE_TO_FILE_H
#define WRITE_TO_FILE_H

#include <iostream>
#include "TempCoherence.h"
#include "TestSuite.h"
#include <iomanip>
#include <il.h>
#include <ilu.h>
#include <ilut.h>
#include "Framebuffer.h"

class WriteToFile
{
public:
	std::string targetFileName;
	GLubyte* pixelBuffer[800*800*4];
	int xPixels, yPixels;

	void Init(int xPixels_, int yPixels_)
	{
		xPixels = xPixels_;
		yPixels = yPixels_;

//		targetFileName = "../TestDumps/Epsilon_3_0.txt";
//		std::remove(targetFileName.c_str());
//
//		ofstream outStream(targetFileName);
//		if (outStream.is_open())
//		{
////			outStream << "Time \t\tCopy \tExtrap \tMSE \t\t\tMAE \t\t\tPSN \t\t\tl1 \t\t\tl2 \t\t\tl3 \t\t\tl4 \t\t\tl5" << std::endl;
//			outStream.close();
//		}
	}


	void Write(int currentTimestep, TempCoherence &tC, TestSuite &test)
	{
		ofstream outStream(targetFileName, std::ios::app);

		if (outStream.is_open())
		{
			outStream << currentTimestep << "\t\t" << tC.numBlocksCopied << "\t" << tC.numBlocksExtrapolated << std::fixed << std::setprecision(6) << "\t\t" << test.errorMetrics.meanSqrError << "\t\t" << test.errorMetrics.meanAvgErr << "\t\t" << test.errorMetrics.peakSigToNoise << "\t\t" << test.errorMetrics.laplaceMSE << std::endl;
//			outStream << currentTimestep << std::fixed << std::setprecision(6) << "\t\t" << test.changeBetweenFrames.la1 << "\t\t" << test.changeBetweenFrames.la2 << "\t\t" << test.changeBetweenFrames.la3 << "\t\t" << test.changeBetweenFrames.la4 << "\t\t" << test.changeBetweenFrames.la5 << std::endl;
			outStream.close();
		}
		if (currentTimestep == 598)
			getchar();
	}

	void WriteImage(int currentTimestep)
	{
		ILboolean success;
		ILuint imageID;
		ilInit();
		std::string imageName("ImageDumps/Ep_1_0_Time_" + std::to_string(currentTimestep) + ".png");
		ilGenImages(1, &imageID);
		ilBindImage(imageID);

		glReadPixels(0, 0, xPixels, yPixels, GL_RGBA, GL_UNSIGNED_BYTE, pixelBuffer);

		GLenum err = glGetError();

		if (err != GL_NO_ERROR)
		    printf("glError: %s\n", gluErrorString(err));

		success = ilTexImage(xPixels, yPixels, 0, 4, IL_RGBA, IL_UNSIGNED_BYTE, pixelBuffer);

		success = ilSave(IL_PNG, imageName.c_str());

		ilDeleteImages(1, &imageID);

	}

};

#endif