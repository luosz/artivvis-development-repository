#pragma once

#ifndef TransferFunctionView_H
#define TransferFunctionView_H

#include <iostream>
#include <QResizeEvent>
#include <QMenu>
#include <QSharedPointer>
#include <glm/glm.hpp>
#include "graphwidget.h"
#include "node.h"
#include "ControlPoint.h"
#include "ControlEdge.h"
#include "TransferFunctionScene.h"
#include "AbstractGraphicsView.h"

//! [0];
class TransferFunctionView : public AbstractGraphicsView
{
public:
	TransferFunctionView(QWidget *parent = 0) : AbstractGraphicsView(parent)
	{
		QSize size = this->size();
		QGraphicsScene* tfScene = static_cast<QGraphicsScene*>(new TransferFunctionScene(this));
		this->setScene(tfScene);
		std::cout << "TransferFunctionView size " << size.width() << " " << size.height() << "\t";
		scene()->setSceneRect(0, 0, size.width(), size.height());
		QRectF rect = this->sceneRect();
		std::cout << "sceneRect " << rect.left() << " " << rect.top()<<" "<<rect.width()<<" "<<rect.height() << std::endl;
		scene()->clear();

#ifndef NOT_USED_BY_VOLUME_RENDERER
		transfer_function = NULL;
#endif // NOT_USED_BY_VOLUME_RENDERER

		is_ma_optimizer_enable = false;
		is_luo_optimizer_enable = false;
	}

	virtual ~TransferFunctionView()
	{
	}

	void setTransferFunction(int numIntensities, std::vector<glm::vec4> colors, std::vector<float> intensities)
	{
		this->numIntensities = numIntensities;
		this->colors = colors;
		this->intensities = intensities;
		updateTransferFunctionFromView_and_drawTransferFunction(true);
	}

	void getTransferFunction(int &numIntensities, std::vector<glm::vec4> &colors, std::vector<float> &intensities)
	{
		numIntensities = this->numIntensities;
		colors = this->colors;
		intensities = this->intensities;
	}

	void updateTransferFunctionFromView_and_drawTransferFunction(bool upate_origColors = false)
	{
		updateTransferFunctionFromView(upate_origColors);
		draw();
	}

	virtual void draw()
	{
		QSize size = this->size();
		scene()->clear();
		ControlPoint *node0 = NULL;
		for (int i=0; i<numIntensities; i++)
		{
			glm::vec4 c = colors[i];
			ControlPoint* node1 = new ControlPoint(this, i, QColor(int(c.r * 255), int(c.g * 255), int(c.b * 255)));
			scene()->addItem(node1);
			node1->setPos(intensities[i] * size.width(), (1 - c.a) * size.height());
			if (i>=1)
			{
				scene()->addItem(new ControlEdge(static_cast<Node*>(node0), static_cast<Node*>(node1)));
			}
			node0 = node1;
		}
	}

	virtual void removeControlPoint(int index)
	{
		std::cout << "removeControlPoint size before " << numIntensities << " after ";
		colors.erase(colors.begin() + index);
		intensities.erase(intensities.begin() + index);
		numIntensities = intensities.size();
		std::cout << numIntensities << std::endl;
		updateTransferFunctionFromView_and_drawTransferFunction(true);
	}

	virtual void moveControlPoint(int index, double intensity, double opacity)
	{
		intensity = intensity < 0 ? 0 : intensity;
		intensity = intensity > 1 ? 1 : intensity;
		opacity = opacity < 0 ? 0 : opacity;
		opacity = opacity > 1 ? 1 : opacity;
		intensities[index] = intensity;
		colors[index].a = opacity;
		for (int i = 0; i < intensities.size(); i++)
		{
			if (i < index && intensities[index] < intensities[i])
			{
				intensities[i] = intensities[index];
			}
			if (i > index && intensities[index] > intensities[i])
			{
				intensities[i] = intensities[index];
			}
		}
		updateTransferFunctionFromView_and_drawTransferFunction(true);
	}

	virtual void addControlPoint(double intensity, double opacity)
	{
		intensity = intensity < 0 ? 0 : intensity;
		intensity = intensity > 1 ? 1 : intensity;
		opacity = opacity < 0 ? 0 : opacity;
		opacity = opacity > 1 ? 1 : opacity;
		int size = intensities.size();
		int i = 0;
		while (i < size && intensities[i] < intensity)
		{
			i++;
		}
		if (i == 0)
		{
			intensities.insert(intensities.begin(), intensity);
			glm::vec4 c = glm::vec4(colors[0].r, colors[0].g, colors[0].b, opacity);
			colors.insert(colors.begin(), c);
		} 
		else
		{
			if (i >= size)
			{
				if (i > size)
				{
					std::cout << "Error: index out of range. i=" << i << " size=" << size << std::endl;
				}
				intensities.push_back(intensity);
				glm::vec4 c = glm::vec4(colors[size - 1].r, colors[size - 1].g, colors[size - 1].b, opacity);
				colors.push_back(c);
			}
			else
			{
				glm::vec4 c0 = colors[i - 1];
				glm::vec4 c1 = colors[i];
				double intensity0 = intensities[i - 1];
				double intensity1 = intensities[i];
				double fraction = (intensity - intensity0) / (intensity1 - intensity0);
				double r = c0.r + (c1.r - c0.r) * fraction;
				double g = c0.g + (c1.g - c0.g) * fraction;
				double b = c0.b + (c1.b - c0.b) * fraction;
				glm::vec4 c = glm::vec4(r, g, b, opacity);
				intensities.insert(intensities.begin() + i, intensity);
				colors.insert(colors.begin() + i, c);
			}
		}
		numIntensities = intensities.size();
		updateTransferFunctionFromView_and_drawTransferFunction(true);
	}

	virtual void setSelectedIndex(int index)
	{
		this->selectedIndex = index;
	}

	virtual void updateTransferFunctionFromView(bool upate_origColors = false)
	{
#ifndef NOT_USED_BY_VOLUME_RENDERER
		if (transfer_function)
		{
			//transfer_function->optimizeIntensity = true;
			transfer_function->numIntensities = numIntensities;
			transfer_function->intensities = intensities;
			transfer_function->colors = colors;
			if (upate_origColors)
			{
				transfer_function->origColors = colors;
			}
			transfer_function->LoadLookup(transfer_function->currentColorTable);
		}
#endif // NOT_USED_BY_VOLUME_RENDERER
	}

	virtual void updateViewFromTransferFunction()
	{
#ifndef NOT_USED_BY_VOLUME_RENDERER
		if (transfer_function)
		{
			//transfer_function->optimizeIntensity = false;
			numIntensities = transfer_function->numIntensities;
			intensities = transfer_function->intensities;
			colors = transfer_function->colors;
			draw();
		}
#endif // NOT_USED_BY_VOLUME_RENDERER
	}

	virtual void optimizeForIntensity(int index)
	{
#ifndef NOT_USED_BY_VOLUME_RENDERER
		// optimize for selected intensity
		if (transfer_function)
		{
			updateTransferFunctionFromView();
//			transfer_function->targetIntensity = intensities[index];
//			transfer_function->intensityOptimizerV2->Optimize(transfer_function->targetIntensity);
//			transfer_function->LoadLookup(transfer_function->currentColorTable);
			updateViewFromTransferFunction();
		}
		else
		{
			std::cout << "Error in optimizeForIntensity: transfer_function is NULL." << std::endl;
		}
#endif // NOT_USED_BY_VOLUME_RENDERER
	}

	virtual bool isMaOptimizerEnable()
	{
		return is_ma_optimizer_enable;
	}

	virtual bool isLuoOptimizerEnable()
	{
		return is_luo_optimizer_enable;
	}

	virtual void changeControlPointColor(int index, QColor color)
	{
		glm::vec4 c(color.red() / 255.f, color.green() / 255.f, color.blue() / 255.f, colors[index].a);
		colors[index] = c;
		updateTransferFunctionFromView_and_drawTransferFunction(true);
	}

	int getSelectedIndex()
	{
		return selectedIndex;
	}

	void distributeVertically()
	{
		int size = colors.size();
		double min = 1, max = 0;
		for (int i = 0; i < size; i++)
		{
			if (colors[i].a < min)
			{
				min = colors[i].a;
			}
			if (colors[i].a > max)
			{
				max = colors[i].a;
			}
		}
		double range = max - min;
		double interval = range / (size - 1);
		for (int i = 0; i < size; i++)
		{
			colors[i].a = min + i * interval;
		}
		updateTransferFunctionFromView_and_drawTransferFunction(true);
	}

	void distrubuteHorizontally()
	{
		int size = intensities.size();
		double range = intensities[size - 1] - intensities[0];
		double interval = range / (size - 1);
		for (int i = 0; i < size; i++)
		{
			intensities[i] = intensities[0] + i * interval;
		}
		updateTransferFunctionFromView_and_drawTransferFunction(true);
	}

	void makeDiagonal()
	{
		int size = colors.size();
		for (int i = 0; i < size; i++)
		{
			colors[i].a = intensities[i];
		}
		updateTransferFunctionFromView_and_drawTransferFunction(true);
	}

	void makeFlat(double opacity = 0.1)
	{
		int size = colors.size();
		for (int i = 0; i < size; i++)
		{
			colors[i].a = opacity;
		}
		updateTransferFunctionFromView_and_drawTransferFunction(true);
	}

	void makeRamp(double opacity = 0.1)
	{
		int size = colors.size();
		for (int i = 0; i < size; i++)
		{
			if (i == 0 || i == size - 1)
			{
				colors[i].a = 0;
			}
			else
			{
				colors[i].a = opacity;
			}
		}
		updateTransferFunctionFromView_and_drawTransferFunction(true);
	}

protected:
	virtual void resizeEvent(QResizeEvent * event)
	{
		QSize size = event->size();
		std::cout << "TransferFunctionView::resizeEvent size " << size.width() << " " << size.height() << "\t";
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
	int numIntensities;
	std::vector<glm::vec4> colors;
	std::vector<float> intensities;
	int selectedIndex;
	bool is_ma_optimizer_enable;
	bool is_luo_optimizer_enable;

#ifndef NOT_USED_BY_VOLUME_RENDERER
public:
	TransferFunction *transfer_function;
#endif // NOT_USED_BY_VOLUME_RENDERER
};
//! [0]

#endif // TransferFunctionView_H
