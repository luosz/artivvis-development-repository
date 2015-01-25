#pragma once

#ifndef CONTROL_POINT_H
#define CONTROL_POINT_H

#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QPainter>
#include <QStyleOption>
#include <QColor>
#include <QMenu>
#include <QPoint>
#include <QColorDialog>
#include <iostream>
#include "edge.h"
#include "node.h"
#include "AbstractGraphicsView.h"

class ControlPoint : public Node
{
public:
	ControlPoint(AbstractGraphicsView *graphWidget, int index, QColor color = QColor(Qt::yellow)) : Node(static_cast<GraphWidget*>(graphWidget))
	{
		this->_color = color;
		this->_index = index;
		view = graphWidget;
	}

	virtual ~ControlPoint()
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
		painter->drawEllipse(-7, -7, 20, 20);

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

		painter->setPen(QPen(Qt::black, 0));
		painter->drawEllipse(-10, -10, 20, 20);
	}

protected:
	virtual void mousePressEvent(QGraphicsSceneMouseEvent * event)
	{
		Node::mousePressEvent(event);
		view->setSelectedIndex(_index);
		event->accept();
	}

	virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent * event)
	{
		Node::mouseReleaseEvent(event);
		if (!event->isAccepted() && event->button() == Qt::LeftButton)
		{
			QRectF size = view->sceneRect();
			QPointF pos = event->scenePos();
			qreal intensity = pos.x() / size.width();
			qreal opacity = 1 - pos.y() / size.height();
			view->moveControlPoint(_index, intensity, opacity);
			event->accept();
		}
	}

	virtual void contextMenuEvent(QGraphicsSceneContextMenuEvent * event)
	{
		QGraphicsItem::contextMenuEvent(event);
		QMenu menu;
		QAction *removeAction = menu.addAction("Remove");
		QAction *changeAction = menu.addAction("Change color...");
		QAction *optimizeForIntensityAction = menu.addAction("Optimize for intensity");
		QAction *optimizeAction = menu.addAction("Optimize transfer function");
		QAction *selectedAction = menu.exec(event->screenPos());
		if (selectedAction == removeAction)
		{
			view->removeControlPoint(_index);
		}
		else
		{
			if (selectedAction == changeAction)
			{
				// change control point color
				QColor c = QColorDialog::getColor(_color, view);
				if (c.isValid())
				{
					_color = c;
					view->changeControlPointColor(_index, _color);
				}
			}
			else
			{
				if (selectedAction == optimizeForIntensityAction)
				{
					// optimize for intensity
					view->optimizeForIntensity(_index);
				}
				else
				{
					if (selectedAction == optimizeAction)
					{
						// optimize the transfer function
						view->optimize();
					}
				}
			}
		}
		event->accept();
	}

protected:
	QColor _color;
	int _index;
	AbstractGraphicsView *view;
};

#endif // ControlPoint_H
