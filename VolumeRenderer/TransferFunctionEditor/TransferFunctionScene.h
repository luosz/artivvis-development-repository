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
		if (!event->isAccepted() && event->button() == Qt::MouseButton::LeftButton)
		{
			auto view = static_cast<AbstractGraphicsView*>(this->parent());
			auto size = view->sceneRect();
			auto pos = event->scenePos();
			auto intensity = pos.x() / size.width();
			auto opacity = 1 - pos.y() / size.height();
			view->addControlPoint(intensity, opacity);
			event->accept();
		}
	}
};

#endif // TransferFunctionScene_H
