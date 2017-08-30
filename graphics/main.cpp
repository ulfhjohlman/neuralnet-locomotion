#include "stdafx.h"
#include "GraphicsEngine.h"
#include "PlotWidget.h"
#include "../practice/odeinttests.h"

#include <QtWidgets/QApplication>
#include <QtPrintSupport/QtPrintSupport>

struct push_back_state_and_time
{
	std::vector<double>& m_x;
	std::vector<double>& m_z;

	PlotWidget * plot_window;

	push_back_state_and_time(std::vector< double > &x_, std::vector< double > &z_, PlotWidget * plow_window_)
		: m_x(x_), m_z(z_), plot_window( plow_window_ ) {
		m_x.reserve(1012);
		m_z.reserve(1012);
	}

	void operator()(const state_type &x, double t)
	{
		m_x.push_back(x[0]);
		m_z.push_back(x[2]);
	}
};


int main(int argc, char *argv[])
{
	QApplication application(argc, argv);


	PlotWidget plot_window;
	//plot_window.setStyle(PlotWidget::dark);
	plot_window.show();
	std::vector<double> x1;
	std::vector<double> z1;
	push_back_state_and_time states(x1, z1, &plot_window);

	state_type x0 = { 10.0 , 1.0 , 1.0 }; // initial conditions
	integrate(lorenz, x0, 0.0, 25.0, 0.1, states);
	plot_window.plot(x1.data(), z1.data(), z1.size(), "asd");
	
	return application.exec();
}
