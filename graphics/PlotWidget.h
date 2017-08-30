#pragma once

#include <QtWidgets/QWidget>
#include <QtPrintSupport/QtPrintSupport>
#include <QInputDialog>
#include <QMessageBox>
#include <QEvent>
#include <QKeyEvent>
#include <thread>

#include "ui_PlotWidget.h"

#include "../lib/qcustomplot/qcustomplot.h"
#include "QVector"

class PlotWidget : public QWidget
{
	Q_OBJECT

public:
	PlotWidget(QWidget* parent = Q_NULLPTR);
	~PlotWidget();

private slots:
	void titleDoubleClick(QMouseEvent *event);
	void axisLabelDoubleClick(QCPAxis* axis, QCPAxis::SelectablePart part);
	void legendDoubleClick(QCPLegend* legend, QCPAbstractLegendItem* item);
	void selectionChanged();
	void mousePress();
	void keyPressEvent(QKeyEvent * event);
	void mouseWheel();
	void removeSelectedGraph();
	void removeAllGraphs();
	void contextMenuRequest(QPoint pos);
	void moveLegend();
	void graphClicked(QCPAbstractPlottable *plottable, int dataIndex);
	
	void save();

	void addRandomGraph();
public:
	void setMarker(int graph_index, QCPScatterStyle::ScatterShape marker, bool replot = true);
	void setLine(int graph_index, QCPGraph::LineStyle line, bool replot = true);
	void setColor(int graph_index, QColor color, bool replot = true);
	QCustomPlot* getPlot() { return customPlot; }

	template<typename T>
	void plot( T *  x_,  T *  y_, size_t size, const char* name = "") {
		if (size < 1)
			return;

		//How the fuck do I move content not copy, i.e setting qcpdata = x_
		QVector<T> x(size), y(size);
		std::copy(&x_[0], &x_[size], x.begin() );
		std::copy(&y_[0], &y_[size], y.begin() );

		customPlot->addGraph();
		customPlot->graph()->setName(QString(name));
		customPlot->graph()->setData(x, y, true);
		customPlot->graph()->setLineStyle(QCPGraph::LineStyle::lsLine);

		QPen graphPen;
		graphPen.setColor(QColor("cyan"));
		//graphPen.setWidthF(1);
		customPlot->graph()->setPen(graphPen);

		customPlot->graph()->rescaleAxes(false);
		customPlot->rescaleAxes(true);
		customPlot->replot();
	}
	enum PlotTheme {
		dark,
		normal,
	};
	void setStyle(PlotTheme ps);
private:
	Ui::Form ui;
	QCustomPlot* customPlot;
};