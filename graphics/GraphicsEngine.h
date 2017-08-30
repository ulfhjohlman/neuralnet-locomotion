#pragma once

#include <QtWidgets/QMainWindow>
#include <QtWidgets/QSizePolicy>
#include <QMessageBox>
#include <QEvent>
#include <QKeyEvent>

#include "ui_GraphicsEngine.h"

//dummy class so far
class GraphicsEngine : public QMainWindow 
{
	Q_OBJECT

public:
	GraphicsEngine(QWidget *parent = Q_NULLPTR);
	void keyPressEvent(QKeyEvent * event);

private:
	Ui::GraphicsEngineClass ui;
};
