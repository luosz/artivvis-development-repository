#include "transferfunctioneditor.h"
#include <QtWidgets/QApplication>

int main_old(int argc, char *argv[])
{
	QApplication a(argc, argv);
	TransferFunctionEditor w;
	w.show();
	return a.exec();
}
