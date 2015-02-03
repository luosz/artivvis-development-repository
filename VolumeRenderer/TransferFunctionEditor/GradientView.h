#ifndef GradientView_h
#define GradientView_h

#pragma once

#include "MyGraphicsView.h"

class GradientView : public MyGraphicsView
{
public:

	GradientView(std::string name = "", QWidget *parent = 0) : MyGraphicsView(parent)
	{
		setName(name);
		QSize size = this->size();
		std::cout << _name << " HistogramView size " << size.width() << " " << size.height() << "\t";
		scene()->setSceneRect(0, 0, size.width(), size.height());
		QRectF rect = this->sceneRect();
		std::cout << "sceneRect " << rect.left() << " " << rect.top() << " " << rect.width() << " " << rect.height() << std::endl;
		scene()->clear();
	}

	virtual ~GradientView()
	{
	}

	virtual void draw(){}
};

#endif // GradientView_h
