/**
**  This file is part of OCTproZ.
**  OCTproZ is an open source software for processig of optical
**  coherence tomography (OCT) raw data.
**  Copyright (C) 2019-2022 Miroslav Zabic
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

#include "extensionmanager.h"


ExtensionManager::ExtensionManager(QObject *parent) : QObject(parent)
{
}

ExtensionManager::~ExtensionManager()
{
	qDeleteAll(this->extensions);
	this->extensions.clear();
	this->extensionNames.clear();
}

void ExtensionManager::addExtension(Extension* extension) {
	if (extension != nullptr) {
		if (!extensions.contains(extension)) {
			this->extensions.append(extension);
			this->extensionNames.append(extension->getName());
			this->extensionNames.last().detach(); //force deep copy of appended extension name to avoid possible problems if plugin lives at some point in a thread
		}
	}
}

Extension* ExtensionManager::getExtensionByName(QString name) {
	int index = this->extensionNames.indexOf(name);
	return index == -1 ? nullptr : this->extensions.at(index);
}
