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

#include "systemchooser.h"


SystemChooser::SystemChooser() : QDialog()
{
	this->selectedSystem = "";
	layout = new QGridLayout(this);
	label = new QLabel("The following systems are available:");
	layout->addWidget(label);
	listView = new QListWidget();
	listView->setSelectionMode(QAbstractItemView::SingleSelection);
	layout->addWidget(listView);
	pushButton_ok = new QPushButton("Select");
	layout->addWidget(pushButton_ok);
	connect(this->pushButton_ok, &QPushButton::clicked, this, &SystemChooser::slot_select);

}

SystemChooser::~SystemChooser(){
	delete this->layout;
	delete this->label;
	delete this->listView;
	delete this->pushButton_ok;
}

void SystemChooser::populate(QList<QString> systems)
{
	this->listView->clear();
	for(auto system : systems){
		this->listView->addItem(new QListWidgetItem(system));
	}
}

QString SystemChooser::selectSystem(QList<QString> systems)
{
	this->populate(systems);
	this->exec();
	return selectedSystem;
}

void SystemChooser::slot_select()
{
	if(!listView->selectedItems().isEmpty())
	{
		QListWidgetItem* selectedItem = listView->selectedItems()[0];
		this->selectedSystem = selectedItem->text();

	}
	close();
	listView->clear();
}
