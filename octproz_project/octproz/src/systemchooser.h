/**
**  This file is part of OCTproZ.
**  OCTproZ is an open source software for processig of optical
**  coherence tomography (OCT) raw data.
**  Copyright (C) 2019-2020 Miroslav Zabic
**
**  OCTproZ is free software: you can redistribute it and/or modify
**  it under the terms of the GNU General Public License as published by
**  the Free Software Foundation, either version 3 of the License, or
**  (at your option) any later version.
**
**  This program is distributed in the hope that it will be useful,
**  but WITHOUT ANY WARRANTY; without even the implied warranty of
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
**  GNU General Public License for more details.
**
**  You should have received a copy of the GNU General Public License
**  along with this program. If not, see http://www.gnu.org/licenses/.
**
****
** Author:	Miroslav Zabic
** Contact:	zabic
**			at
**			iqo.uni-hannover.de
****
**/

#ifndef SYSTEMCHOOSER_H
#define SYSTEMCHOOSER_H

#include <QObject>
#include <QDialog>
#include <QListWidget>
#include <QLabel>
#include <QGridLayout>
#include <QLayout>
#include <QPushButton>


class SystemChooser : public QDialog
{
	Q_OBJECT
public:
	SystemChooser();
	~SystemChooser();

	QString selectSystem(QList<QString> systems);

private:
	QString selectedSystem;
	QLayout* layout;
	QListWidget* listView;
	QLabel* label;
	QPushButton* pushButton_ok;

	void populate(QList<QString> systems);

public slots:
	void onOkClicked();
	void onDoubleClicked(QListWidgetItem* item);
};

#endif //SYSTEMCHOOSER_H
