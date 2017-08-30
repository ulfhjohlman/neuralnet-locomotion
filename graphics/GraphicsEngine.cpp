#include "stdafx.h"
#include "GraphicsEngine.h"

GraphicsEngine::GraphicsEngine(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
}

void GraphicsEngine::keyPressEvent(QKeyEvent * event)
{
	if(event->key() == Qt::Key_Escape)
	{
		QMessageBox msgBox;
		msgBox.setText("The window is about to be closed.");
		msgBox.setInformativeText("Do you want to close?");
		msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
		msgBox.setDefaultButton(QMessageBox::No);
		int ret = msgBox.exec();


		if( ret == QMessageBox::Yes )
			close();
	}
	else
		QWidget::keyPressEvent(event);
}
