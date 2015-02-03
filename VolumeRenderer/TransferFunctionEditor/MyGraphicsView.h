#ifndef MyGraphicsView_h
#define MyGraphicsView_h

#pragma once

#include <iostream>
#include <string>
#include <QResizeEvent>
#include "graphwidget.h"

class MyGraphicsView : public GraphWidget
{
public:

	MyGraphicsView(QWidget *parent = 0) : GraphWidget(parent)
	{
	}

	virtual ~MyGraphicsView()
	{
	}

	virtual void draw() = 0;

	virtual void setName(std::string name)
	{
		this->_name = name;
	}

	std::string _name;

protected:

	virtual void resizeEvent(QResizeEvent * event)
	{
		QSize size = event->size();
		std::cout << "MyGraphicsView::resizeEvent size " << size.width() << " " << size.height() << "\t";
		scene()->setSceneRect(0, 0, size.width(), size.height());
		QRectF rect = this->sceneRect();
		std::cout << "sceneRect " << rect.left() << " " << rect.top() << " " << rect.width() << " " << rect.height() << std::endl;
		draw();
	}

	virtual void drawBackground(QPainter *painter, const QRectF &rect)
	{
		QRectF sceneRect = this->sceneRect();
		painter->drawRect(sceneRect);
	}

	virtual void timerEvent(QTimerEvent *event){}
};

#endif // MyGraphicsView_h
