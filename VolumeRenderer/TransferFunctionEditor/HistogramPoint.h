#pragma once

#ifndef HistogramPoint_h
#define HistogramPoint_h

#include "node.h"

class HistogramPoint : public Node
{
public:
	HistogramPoint() : Node(NULL)
	{
		_index = -1;
		_color = QColor(Qt::white);
	}

	HistogramPoint(GraphWidget *graphWidget, int index, QColor color = QColor(Qt::yellow)) : Node(static_cast<GraphWidget*>(graphWidget))
	{
		this->_color = color;
		this->_index = index;
		setFlag(ItemIsMovable, false);
	}

	virtual ~HistogramPoint()
	{
	}

	int index()
	{
		return this->_index;
	}

	virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
	{
		painter->setPen(Qt::NoPen);
		painter->setBrush(Qt::darkGray);
		painter->drawEllipse(-5, -5, 10, 10);

		QRadialGradient gradient(-3, -3, 10);
		if (option->state & QStyle::State_Sunken) {
			gradient.setCenter(3, 3);
			gradient.setFocalPoint(3, 3);
			gradient.setColorAt(1, QColor(_color).light(120));
			gradient.setColorAt(0, QColor(_color.darker()).light(120));
		}
		else {
			gradient.setColorAt(0, _color);
			gradient.setColorAt(1, _color.darker());
		}
		painter->setBrush(gradient);

		//// draw node boundary
		//painter->setPen(QPen(Qt::black, 0));

		painter->drawEllipse(-5, -5, 10, 10);
	}

protected:
	QColor _color;
	int _index;
};

#endif
