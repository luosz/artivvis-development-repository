#include "ErrorEvaluator.h"

void ErrorEvaluator::Init(int screenWidth, int screenHeight)
{
	xPixels = screenWidth;
	yPixels = screenHeight;
	numPixels = xPixels * yPixels;
}