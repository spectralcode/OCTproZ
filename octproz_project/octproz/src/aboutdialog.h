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
**			spectralcode.de
****
**/

#ifndef ABOUTDIALOG_H
#define ABOUTDIALOG_H

#include <QDialog>
#include <QMap>

class QTabWidget;
class QVBoxLayout;
class QHBoxLayout;
class QListWidget;
class QTextBrowser;

struct ComponentInfo {
	QString name;
	QString url;
	QString licensePath;
};

class AboutDialog : public QDialog
{
	Q_OBJECT
public:
	explicit AboutDialog(QWidget *parent = nullptr);
	~AboutDialog();

private:
	void setupWindowProperties();
	QVBoxLayout* createLogoLayout();
	void setupTabs(QTabWidget *tabWidget);
	void addAboutTab(QTabWidget *tabWidget);
	void addLicenseTab(QTabWidget *tabWidget);
	void addCreditsTab(QTabWidget *tabWidget);
	void addThirdPartyTab(QTabWidget *tabWidget);
	QHBoxLayout* createButtonLayout();
	void setupConnections(QTabWidget *tabWidget);

	// New methods for third-party components tab
	QList<ComponentInfo> getThirdPartyComponents();
	QString loadLicenseText(const QString& path);
	void updateComponentDetails(QTextBrowser* browser, const ComponentInfo& component);

signals:
	void easterEgg();

public slots:
};

#endif // ABOUTDIALOG_H
