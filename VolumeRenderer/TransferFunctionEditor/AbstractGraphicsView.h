#pragma once

#ifndef AbstractGraphicsView_h
#define AbstractGraphicsView_h

#include "MyGraphicsView.h"

class AbstractGraphicsView : public MyGraphicsView
{
public:
	AbstractGraphicsView(QWidget *parent = 0) : MyGraphicsView(parent)
	{
	}

	virtual ~AbstractGraphicsView()
	{
	}

	// interface for use in TransferFunctionView
	virtual void optimizeForIntensity(int index){}
	virtual void optimize(){}
	virtual void removeControlPoint(int index){}
	virtual void moveControlPoint(int index, double intensity, double opacity){}
	virtual void addControlPoint(double intensity, double opacity){}
	virtual void setSelectedIndex(int index){}
	virtual void changeControlPointColor(int index, QColor color){}
	virtual void updateTransferFunctionFromView(bool upate_origColors = true){}
	virtual void updateViewFromTransferFunction(){}
	virtual bool isMaOptimizerEnable(){ return false; }
	virtual bool isLuoOptimizerEnable(){ return false; }
	virtual void setVisibilityHistogram(const std::vector<float> &visibilities, const std::vector<int> &numVis)
	{
		std::cout << "AbstractGraphicsView::setVisibilityHistogram is not implemented." << std::endl;
	}
};

#endif // AbstractGraphicsView_h
