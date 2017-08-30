#pragma once
#include <QtWidgets/QWidget>
#include <QtPrintSupport/QtPrintSupport>
#include "ui_ViewWindow.h"

class ViewWindow :
	public QWidget
{
	Q_OBJECT

public:
	ViewWindow(QWidget *parent);
	~ViewWindow();
private:
	Ui::Form ui;
};

