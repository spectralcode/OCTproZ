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

#ifndef MESSAGECONSOLE_H
#define MESSAGECONSOLE_H

#include <QTextEdit>
#include <QGridLayout>
#include <QDateTime>
#include <QDockWidget>
#include <QWidget>
#include <QMenu>
#include <QContextMenuEvent>
#include <QCoreApplication>

#define MAX_MESSAGES 512


class MessageConsole : public QWidget
{
	Q_OBJECT
public:
	explicit MessageConsole(QWidget *parent = nullptr);
	~MessageConsole();

	QDockWidget* getDock(){return this->dock;}

private:
	QTextEdit* textEdit;
	QGridLayout* gridLayout;
	QDockWidget* dock;
	QPoint mousePos;
	QVector<QString> messages;
	int messagesIndex;

	QString addStringToMessageBuffer(QString message);
	void contextMenuEvent(QContextMenuEvent* event);

signals:
	void error(QString);
	void info(QString);

public slots:
	void displayInfo(QString info);
	void displayError(QString error);
};

#endif // MESSAGECONSOLE_H
