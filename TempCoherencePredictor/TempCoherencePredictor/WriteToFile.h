#ifndef WRITE_TO_FILE_H
#define WRITE_TO_FILE_H

#include <iostream>
#include "TempCoherence.h"
#include "TestSuite.h"
#include <iomanip>

class WriteToFile
{
public:
	std::string targetFileName;

	void Init()
	{
		targetFileName = "../TestDumps/blah.txt";
		std::remove(targetFileName.c_str());

		ofstream outStream(targetFileName);
		if (outStream.is_open())
		{
//			outStream << "Time \t\tCopy \tExtrap \tMSE \t\t\tMAE \t\t\tPSN \t\t\tl1 \t\t\tl2 \t\t\tl3 \t\t\tl4 \t\t\tl5" << std::endl;
			outStream.close();
		}
	}
	void Write(int currentTimestep, TempCoherence &tC, TestSuite &test)
	{
		ofstream outStream(targetFileName, std::ios::app);

		if (outStream.is_open())
		{
//			outStream << currentTimestep << "\t\t" << tC.numBlocksCopied << "\t" << tC.numBlocksExtrapolated << std::fixed << std::setprecision(6) << "\t\t" << test.errorMetrics.meanSqrError << "\t\t" << test.errorMetrics.meanAvgErr << "\t\t" << test.errorMetrics.peakSigToNoise << std::endl;
			outStream << currentTimestep << std::fixed << std::setprecision(6) << "\t\t" << test.changeBetweenFrames.la1 << "\t\t" << test.changeBetweenFrames.la2 << "\t\t" << test.changeBetweenFrames.la3 << "\t\t" << test.changeBetweenFrames.la4 << "\t\t" << test.changeBetweenFrames.la5 << std::endl;
			outStream.close();
		}
		if (currentTimestep == 598)
			getchar();
	}
};

#endif