#include "stdafx.h"
#include "PlotWidget.h"

PlotWidget::PlotWidget(QWidget* parent) : QWidget(parent), customPlot(Q_NULLPTR) {
	ui.setupUi(this);
	//setWindowFlags(Qt::CustomizeWindowHint);

	customPlot = new QCustomPlot(this);
	this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

	ui.gridLayout->addWidget(customPlot, 0, 0, 1, 1);

	customPlot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectAxes |
		QCP::iSelectLegend | QCP::iSelectPlottables);

	customPlot->plotLayout()->insertRow(0);
	QCPTextElement *title = new QCPTextElement(customPlot, "title", QFont("sans", 15, QFont::Bold));
	title->setTextColor(QColor(255, 255, 255));
	customPlot->plotLayout()->addElement(0, 0, title);

	QFont axisFont("sans", 12, QFont::Normal);
	customPlot->xAxis->setLabel("x Axis");
	customPlot->xAxis->setLabelFont(axisFont);

	customPlot->yAxis->setLabel("y Axis");
	customPlot->yAxis->setLabelFont(axisFont);
	
	
	customPlot->legend->setVisible(true);
	QFont legendFont = font();
	legendFont.setPointSize(10);
	customPlot->legend->setFont(legendFont);
	customPlot->legend->setSelectedFont(legendFont);
	customPlot->legend->setSelectableParts(QCPLegend::spItems); // legend box shall not be selectable, only legend items

	customPlot->rescaleAxes();

	// connect slot that ties some axis selections together (especially opposite axes):
	connect(customPlot, SIGNAL(selectionChangedByUser()), this, SLOT(selectionChanged()));
	// connect slots that takes care that when an axis is selected, only that direction can be dragged and zoomed:
	connect(customPlot, SIGNAL(mousePress(QMouseEvent*)), this, SLOT(mousePress()));
	connect(customPlot, SIGNAL(mouseWheel(QWheelEvent*)), this, SLOT(mouseWheel()));

	//connect(this, SIGNAL(keyPress(QKeyEvent*)), this, SLOT(keyPress()));

	// make bottom and left axes transfer their ranges to top and right axes:
	connect(customPlot->xAxis, SIGNAL(rangeChanged(QCPRange)), customPlot->xAxis2, SLOT(setRange(QCPRange)));
	connect(customPlot->yAxis, SIGNAL(rangeChanged(QCPRange)), customPlot->yAxis2, SLOT(setRange(QCPRange)));

	// connect some interaction slots:
	connect(customPlot, SIGNAL(axisDoubleClick(QCPAxis*, QCPAxis::SelectablePart, QMouseEvent*)), this, SLOT(axisLabelDoubleClick(QCPAxis*, QCPAxis::SelectablePart)));
	connect(customPlot, SIGNAL(legendDoubleClick(QCPLegend*, QCPAbstractLegendItem*, QMouseEvent*)), this, SLOT(legendDoubleClick(QCPLegend*, QCPAbstractLegendItem*)));
	connect(title, SIGNAL(doubleClicked(QMouseEvent*)), this, SLOT(titleDoubleClick(QMouseEvent*)));

	// connect slot that shows a message in the status bar when a graph is clicked:
	connect(customPlot, SIGNAL(plottableClick(QCPAbstractPlottable*, int, QMouseEvent*)), this, SLOT(graphClicked(QCPAbstractPlottable*, int)));

	// setup policy and connect slot for context menu popup:
	customPlot->setContextMenuPolicy(Qt::CustomContextMenu);
	connect(customPlot, SIGNAL(customContextMenuRequested(QPoint)), this, SLOT(contextMenuRequest(QPoint)));

	customPlot->show();
}

void PlotWidget::titleDoubleClick(QMouseEvent *event)
{
	Q_UNUSED(event)
	if (QCPTextElement *title = qobject_cast<QCPTextElement*>(sender())) {
		// Set the plot title by double clicking on it
		bool ok;
		QString newTitle = QInputDialog::getText(this, "QCustomPlot example", "New plot title:", QLineEdit::Normal, title->text(), &ok);
		if (ok) {
			title->setText(newTitle);
			customPlot->replot();
		}
	}
}

PlotWidget::~PlotWidget()
{
	if (customPlot) {
		delete customPlot;
		customPlot = Q_NULLPTR;
	}
}

void PlotWidget::axisLabelDoubleClick(QCPAxis * axis, QCPAxis::SelectablePart part)
{
	// Set an axis label by double clicking on it
	if (part == QCPAxis::spAxisLabel) // only react when the actual axis label is clicked, not tick label or axis backbone
	{
		bool ok;
		QString newLabel = QInputDialog::getText(this, "QCustomPlot example", "New axis label:", QLineEdit::Normal, axis->label(), &ok);
		if (ok) {
			axis->setLabel(newLabel);
			customPlot->replot();
		}
	}
}

void PlotWidget::legendDoubleClick(QCPLegend * legend, QCPAbstractLegendItem * item)
{
	// Rename a graph by double clicking on its legend item
	Q_UNUSED(legend)
	if (item) // only react if item was clicked (user could have clicked on border padding of legend where there is no item, then item is 0)
	{
		QCPPlottableLegendItem *plItem = qobject_cast<QCPPlottableLegendItem*>(item);
		bool ok;
		QString newName = QInputDialog::getText(this, "QCustomPlot example", "New graph name:", QLineEdit::Normal, plItem->plottable()->name(), &ok);
		if (ok)
		{
			plItem->plottable()->setName(newName);
			customPlot->replot();
		}
	}
}

void PlotWidget::selectionChanged()
{
	/*
	normally, axis base line, axis tick labels and axis labels are selectable separately, but we want
	the user only to be able to select the axis as a whole, so we tie the selected states of the tick labels
	and the axis base line together. However, the axis label shall be selectable individually.

	The selection state of the left and right axes shall be synchronized as well as the state of the
	bottom and top axes.

	Further, we want to synchronize the selection of the graphs with the selection state of the respective
	legend item belonging to that graph. So the user can select a graph by either clicking on the graph itself
	or on its legend item.
	*/

	// make top and bottom axes be selected synchronously, and handle axis and tick labels as one selectable object:
	if (customPlot->xAxis->selectedParts().testFlag(QCPAxis::spAxis) || customPlot->xAxis->selectedParts().testFlag(QCPAxis::spTickLabels) ||
		customPlot->xAxis2->selectedParts().testFlag(QCPAxis::spAxis) || customPlot->xAxis2->selectedParts().testFlag(QCPAxis::spTickLabels))
	{
		customPlot->xAxis2->setSelectedParts(QCPAxis::spAxis | QCPAxis::spTickLabels);
		customPlot->xAxis->setSelectedParts(QCPAxis::spAxis | QCPAxis::spTickLabels);
	}
	// make left and right axes be selected synchronously, and handle axis and tick labels as one selectable object:
	if (customPlot->yAxis->selectedParts().testFlag(QCPAxis::spAxis) || customPlot->yAxis->selectedParts().testFlag(QCPAxis::spTickLabels) ||
		customPlot->yAxis2->selectedParts().testFlag(QCPAxis::spAxis) || customPlot->yAxis2->selectedParts().testFlag(QCPAxis::spTickLabels))
	{
		customPlot->yAxis2->setSelectedParts(QCPAxis::spAxis | QCPAxis::spTickLabels);
		customPlot->yAxis->setSelectedParts(QCPAxis::spAxis | QCPAxis::spTickLabels);
	}

	// synchronize selection of graphs with selection of corresponding legend items:
	for (int i = 0; i<customPlot->graphCount(); ++i)
	{
		QCPGraph *graph = customPlot->graph(i);
		QCPPlottableLegendItem *item = customPlot->legend->itemWithPlottable(graph);
		if (item->selected() || graph->selected())
		{
			item->setSelected(true);
			graph->setSelection(QCPDataSelection(graph->data()->dataRange()));
		}
	}
}

void PlotWidget::mousePress()
{
	// if an axis is selected, only allow the direction of that axis to be dragged
	// if no axis is selected, both directions may be dragged

	if (customPlot->xAxis->selectedParts().testFlag(QCPAxis::spAxis))
		customPlot->axisRect()->setRangeDrag(customPlot->xAxis->orientation());
	else if (customPlot->yAxis->selectedParts().testFlag(QCPAxis::spAxis))
		customPlot->axisRect()->setRangeDrag(customPlot->yAxis->orientation());
	else
		customPlot->axisRect()->setRangeDrag(Qt::Horizontal | Qt::Vertical);
}

void PlotWidget::keyPressEvent(QKeyEvent * event)
{
	switch (event->key())
	{
	case Qt::Key_Escape:
		close();
		break;
	default:
		QWidget::keyPressEvent(event);
	}
}

void PlotWidget::mouseWheel()
{
	// if an axis is selected, only allow the direction of that axis to be zoomed
	// if no axis is selected, both directions may be zoomed

	if (customPlot->xAxis->selectedParts().testFlag(QCPAxis::spAxis))
		customPlot->axisRect()->setRangeZoom(customPlot->xAxis->orientation());
	else if (customPlot->yAxis->selectedParts().testFlag(QCPAxis::spAxis))
		customPlot->axisRect()->setRangeZoom(customPlot->yAxis->orientation());
	else
		customPlot->axisRect()->setRangeZoom(Qt::Horizontal | Qt::Vertical);
}

void PlotWidget::addRandomGraph()
{
	int n = 50; // number of points in graph
	double xScale = (rand() / (double)RAND_MAX + 0.5) * 2;
	double yScale = (rand() / (double)RAND_MAX + 0.5) * 2;
	double xOffset = (rand() / (double)RAND_MAX - 0.5) * 4;
	double yOffset = (rand() / (double)RAND_MAX - 0.5) * 10;
	double r1 = (rand() / (double)RAND_MAX - 0.5) * 2;
	double r2 = (rand() / (double)RAND_MAX - 0.5) * 2;
	double r3 = (rand() / (double)RAND_MAX - 0.5) * 2;
	double r4 = (rand() / (double)RAND_MAX - 0.5) * 2;
	QVector<double> x(n), y(n);
	for (int i = 0; i < n; i++)
	{
		x[i] = (i / (double)n - 0.5)*10.0*xScale + xOffset;
		y[i] = (qSin(x[i] * r1 * 5)*qSin(qCos(x[i] * r2)*r4 * 3) + r3*qCos(qSin(x[i])*r4 * 2))*yScale + yOffset;
	}

	customPlot->addGraph();
	customPlot->graph()->setName(QString("New graph %1").arg(customPlot->graphCount() - 1));
	customPlot->graph()->setData(x, y);
	customPlot->graph()->setLineStyle((QCPGraph::LineStyle)(rand() % 5 + 1));
	if (rand() % 100 > 50)
		customPlot->graph()->setScatterStyle(QCPScatterStyle((QCPScatterStyle::ScatterShape)(rand() % 14 + 1)));
	QPen graphPen;
	graphPen.setColor(QColor(rand() % 245 + 10, rand() % 245 + 10, rand() % 245 + 10));
	graphPen.setWidthF(rand() / (double)RAND_MAX * 2 + 1);
	customPlot->graph()->setPen(graphPen);
	customPlot->replot();
}

void PlotWidget::save()
{
	bool ok;
	QString text = QInputDialog::getText(this, tr("QInputDialog::getText()"),
		tr("User name:"), QLineEdit::Normal, ".png", &ok);
	if (ok && !text.isEmpty()) {
		bool save_ok = !customPlot->savePng(text);
		if (save_ok) {
			QMessageBox msgBox;
			msgBox.setText("Unable to save png.");
			msgBox.setInformativeText("QCustomplot failed to save png for some reason.");
			msgBox.setStandardButtons(QMessageBox::Ok);
			msgBox.setDefaultButton(QMessageBox::Ok);
			msgBox.exec();
		}
	}
}

void PlotWidget::setMarker(int graph_index, QCPScatterStyle::ScatterShape marker, bool replot /*= true*/)
{
	customPlot->graph(graph_index)->setScatterStyle(QCPScatterStyle(marker));
	if (replot)
		customPlot->replot();
}

void PlotWidget::setLine(int graph_index, QCPGraph::LineStyle line, bool replot /*= true*/)
{
	customPlot->graph(graph_index)->setLineStyle(line);

	if (replot)
		customPlot->replot();
}

void PlotWidget::setColor(int graph_index, QColor color, bool replot)
{
	QPen graphPen;
	graphPen.setColor(color);
	//graphPen.setWidthF(1);
	customPlot->graph(graph_index)->setPen(graphPen);

	if (replot)
		customPlot->replot();
}

void PlotWidget::setStyle(PlotTheme ps)
{
	if (ps == dark) {
		// set some pens, brushes and backgrounds:
		customPlot->xAxis->setBasePen(QPen(Qt::white, 2));
		customPlot->yAxis->setBasePen(QPen(Qt::white, 2));
		customPlot->xAxis->setTickPen(QPen(Qt::white, 2));
		customPlot->yAxis->setTickPen(QPen(Qt::white, 2));

		customPlot->xAxis->setSubTickPen(QPen(Qt::white, 2));
		customPlot->yAxis->setSubTickPen(QPen(Qt::white, 2));

		customPlot->xAxis->setTickLabelColor(Qt::white);
		customPlot->yAxis->setTickLabelColor(Qt::white);

		customPlot->xAxis->grid()->setPen(QPen(QColor(140, 140, 140), 1, Qt::DotLine));
		customPlot->yAxis->grid()->setPen(QPen(QColor(140, 140, 140), 1, Qt::DotLine));

		customPlot->xAxis->grid()->setSubGridPen(QPen(QColor(80, 80, 80), 1, Qt::DotLine));
		customPlot->yAxis->grid()->setSubGridPen(QPen(QColor(80, 80, 80), 1, Qt::DotLine));

		customPlot->xAxis->grid()->setSubGridVisible(true);
		customPlot->yAxis->grid()->setSubGridVisible(true);

		customPlot->xAxis->grid()->setZeroLinePen(Qt::NoPen);
		customPlot->yAxis->grid()->setZeroLinePen(Qt::NoPen);

		customPlot->xAxis->setUpperEnding(QCPLineEnding::esNone);
		customPlot->yAxis->setUpperEnding(QCPLineEnding::esNone);

		QLinearGradient plotGradient;
		plotGradient.setStart(0, 0);
		plotGradient.setFinalStop(0, 350);
		plotGradient.setColorAt(0, QColor(80, 80, 80));
		plotGradient.setColorAt(1, QColor("#191919"));
		customPlot->setBackground(QColor("#191919"));

		QLinearGradient axisRectGradient;
		axisRectGradient.setStart(0, 0);
		axisRectGradient.setFinalStop(0, 350);
		axisRectGradient.setColorAt(0, QColor(80, 80, 80));
		axisRectGradient.setColorAt(1, QColor(30, 30, 30));

		customPlot->axisRect()->setBackground(QColor(30, 30, 30));

		QFont axisFont("sans", 12, QFont::Normal);
		customPlot->xAxis->setLabelFont(axisFont);
		customPlot->xAxis->setLabelColor(QColor(250, 250, 250));

		customPlot->yAxis->setLabelFont(axisFont);
		customPlot->yAxis->setLabelColor(QColor(250, 250, 250));

		customPlot->legend->setBrush(QBrush(QColor(255, 255, 255, 50)));
		customPlot->legend->setTextColor(QColor(255, 255, 255));
	}
	else if (ps == normal) {

	}

	customPlot->replot();
}

void PlotWidget::removeSelectedGraph()
{
	if (customPlot->selectedGraphs().size() > 0)
	{
		customPlot->removeGraph(customPlot->selectedGraphs().first());
		customPlot->replot();
	}
}

void PlotWidget::removeAllGraphs()
{
	customPlot->clearGraphs();
	customPlot->replot();
}

void PlotWidget::contextMenuRequest(QPoint pos)
{
	QMenu *menu = new QMenu(this);
	menu->setAttribute(Qt::WA_DeleteOnClose);

	if (customPlot->legend->selectTest(pos, false) >= 0) // context menu on legend requested
	{
		menu->addAction("Move to top left", this, SLOT(moveLegend()))->setData((int)(Qt::AlignTop | Qt::AlignLeft));
		menu->addAction("Move to top center", this, SLOT(moveLegend()))->setData((int)(Qt::AlignTop | Qt::AlignHCenter));
		menu->addAction("Move to top right", this, SLOT(moveLegend()))->setData((int)(Qt::AlignTop | Qt::AlignRight));
		menu->addAction("Move to bottom right", this, SLOT(moveLegend()))->setData((int)(Qt::AlignBottom | Qt::AlignRight));
		menu->addAction("Move to bottom left", this, SLOT(moveLegend()))->setData((int)(Qt::AlignBottom | Qt::AlignLeft));
	}
	else  // general context menu on graphs requested
	{
		menu->addAction("Add random graph", this, SLOT(addRandomGraph()));
		menu->addAction("Save plot", this, SLOT(save()));
		if (customPlot->selectedGraphs().size() > 0)
			menu->addAction("Remove selected graph", this, SLOT(removeSelectedGraph()));
		if (customPlot->graphCount() > 0)
			menu->addAction("Remove all graphs", this, SLOT(removeAllGraphs()));
	}

	menu->popup(customPlot->mapToGlobal(pos));
}

void PlotWidget::moveLegend()
{
	if (QAction* contextAction = qobject_cast<QAction*>(sender())) // make sure this slot is really called by a context menu action, so it carries the data we need
	{
		bool ok;
		int dataInt = contextAction->data().toInt(&ok);
		if (ok)
		{
			customPlot->axisRect()->insetLayout()->setInsetAlignment(0, (Qt::Alignment)dataInt);
			customPlot->replot();
		}
	}
}

void PlotWidget::graphClicked(QCPAbstractPlottable *plottable, int dataIndex)
{
	// since we know we only have QCPGraphs in the plot, we can immediately access interface1D()
	// usually it's better to first check whether interface1D() returns non-zero, and only then use it.
	if (!plottable->interface1D()) { //this correct?
		QDialogButtonBox dbb;
		dbb.setWindowIconText(QString("No QCPGraph present."));
		return;
	}

	double dataValue = plottable->interface1D()->dataMainValue(dataIndex);
	QString message = QString("Clicked on graph '%1' at data point #%2 with value %3.").arg(plottable->name()).arg(dataIndex).arg(dataValue);
}
