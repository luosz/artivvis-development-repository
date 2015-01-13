#pragma once

#ifndef TransferFunctionScene_H
#define TransferFunctionScene_H

#include <QMenu>
#include <QGraphicsScene>
#include <QGraphicsSceneContextMenuEvent>
#include "AbstractGraphicsView.h"

class TransferFunctionScene : public QGraphicsScene
{
public:
	TransferFunctionScene(AbstractGraphicsView * parent = 0) : QGraphicsScene(parent)
	{
	}

	TransferFunctionScene(const QRectF & sceneRect, QObject * parent = 0) : QGraphicsScene(sceneRect, parent)
	{}

protected:
	virtual void mousePressEvent(QGraphicsSceneMouseEvent * event)
	{
		QGraphicsScene::mousePressEvent(event);
		if (!event->isAccepted() && event->button() == Qt::LeftButton)
		{
			AbstractGraphicsView* view = static_cast<AbstractGraphicsView*>(this->parent());
			QRectF size = view->sceneRect();
			QPointF pos = event->scenePos();
			qreal intensity = pos.x() / size.width();
			qreal opacity = 1 - pos.y() / size.height();
			view->addControlPoint(intensity, opacity);
			event->accept();
		}
	}
};

#endif // TransferFunctionScene_H
