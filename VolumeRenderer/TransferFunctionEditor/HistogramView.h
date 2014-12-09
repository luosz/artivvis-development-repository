#pragma once

#ifndef HistogramView_h
#define HistogramView_h

#include <iostream>
#include <string>
#include <vector>
#include "node.h"
#include "edge.h"
#include "graphwidget.h"
#include "HistogramPoint.h"
#include "AbstractGraphicsView.h"

class HistogramView : public AbstractGraphicsView
{
public:
	HistogramView(std::string name = "", QWidget *parent = 0) : AbstractGraphicsView(parent)
	{
		setName(name);
		auto size = this->size();
		std::cout << _name << " HistogramView size " << size.width() << " " << size.height() << "\t";
		scene()->setSceneRect(0, 0, size.width(), size.height());
		QRectF rect = this->sceneRect();
		std::cout << "sceneRect " << rect.left() << " " << rect.top() << " " << rect.width() << " " << rect.height() << std::endl;
		scene()->clear();
	}

	virtual ~HistogramView()
	{
	}

	virtual void draw()
	{
		auto size = this->size();
		scene()->clear();
		HistogramPoint *node0 = NULL;
		auto numIntensities = intensities.size();
		for (int i = 0; i < numIntensities; i++)
		{
			QColor color;
			auto gray = i / (float)numIntensities;
			color.setHsvF(gray, gray, gray);
			auto node1 = new HistogramPoint(this, i, color);
			scene()->addItem(node1);
			node1->setPos(intensities[i] * size.width(), (1 - frequencies[i]) * size.height());
			if (i >= 1)
			{
				scene()->addItem(new ControlEdge(static_cast<Node*>(node0), static_cast<Node*>(node1)));
			}
			node0 = node1;
		}
	}

	virtual void setVisibilityHistogram(std::vector<float> &visibilities, std::vector<int> &numVis)
	{
		if (visibilities.size() != numVis.size())
		{
			std::cout << "Error: visibilities and numVis should be the same size." << std::endl;
		}
		intensities.clear();
		frequencies.clear();
		auto size = visibilities.size();
		for (int i = 0; i < visibilities.size(); i++)
		{
			intensities.push_back(i / (float)size);
			frequencies.push_back(visibilities[i] * numVis[i]);
		}
	}

	void setName(std::string name)
	{
		this->_name = name;
	}

protected:

	virtual void resizeEvent(QResizeEvent * event)
	{
		auto size = event->size();
		std::cout << "HistogramView::resizeEvent size " << size.width() << " " << size.height() << "\t";
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

public:
	std::vector<float> intensities;
	std::vector<float> frequencies;
	std::string _name;
};

#endif // HistogramView_h
