/**
**  This file is part of OCTproZ.
**  OCTproZ is an open source software for processig of optical
**  coherence tomography (OCT) raw data.
**  Copyright (C) 2019-2024 Miroslav Zabic
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

#include "messageconsole.h"


MessageConsole::MessageConsole(QWidget *parent) : QWidget(parent){
	this->gridLayout = new QGridLayout(this);
	this->textEdit = new QTextEdit(this);
	this->textEdit->setReadOnly(true);
	this->textEdit->setContextMenuPolicy(Qt::NoContextMenu);
	this->gridLayout->setMargin(0);
	this->gridLayout->addWidget(this->textEdit, 0, 0, 1, 1);
	this->setMinimumWidth(320);
	this->setMinimumHeight(30);
	this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);

	this->messages = QVector<QString>(MAX_MESSAGES);

	for(auto string : this->messages){
		string = "";
	}
	this->messagesIndex = 0;

	this->params.newestMessageAtBottom = false;
	this->params.preferredHeight = 100;
}

MessageConsole::~MessageConsole(){

}

void MessageConsole::setParams(MessageConsoleParams params) {
	this->params = params;
	this->insertNewMessagesAtBottom(this->params.newestMessageAtBottom);
	this->adjustSize();
}

QSize MessageConsole::sizeHint() const {
	return QSize(QWidget::sizeHint().width(), this->params.preferredHeight);
}

//todo: rethink and redo this. there is probably a much better way than recreating the entire message string for every new message
QString MessageConsole::addStringToMessageBuffer(QString message) {
	this->messages[this->messagesIndex] = message+"<br>";
	QString messagesString;

	if(this->params.newestMessageAtBottom){
		for(int i = 0; i < this->messagesIndex+1; i++){
		messagesString = messagesString + (i == messagesIndex ? "<b>"+this->messages.at(i)+"</b>" : this->messages.at(i));
		}
	} else {
		//append all messages to messagesString and make first one bold
		for(int i = this->messagesIndex+MAX_MESSAGES; i > this->messagesIndex; i--){
		messagesString = messagesString + (i == messagesIndex+MAX_MESSAGES ? "<b>"+this->messages.at(i%MAX_MESSAGES)+"</b>" : this->messages.at(i%MAX_MESSAGES));
		}
	}

	this->messagesIndex = (this->messagesIndex+1) % MAX_MESSAGES;
	return messagesString;
}

void MessageConsole::contextMenuEvent(QContextMenuEvent* event) {
	QMenu* menu =this->textEdit->createStandardContextMenu();

	QAction* messageAtBottom = menu->addAction("Newest message at bottom");
	messageAtBottom->setCheckable(true);
	messageAtBottom->setChecked(this->params.newestMessageAtBottom);
	connect(messageAtBottom, &QAction::toggled, this, &MessageConsole::insertNewMessagesAtBottom);

	menu->exec(event->globalPos());
	delete menu;
}

void MessageConsole::refreshMessages() {
	if(this->messagesIndex>1){
		this->messagesIndex--;
	} else {
		return;
	}

	QString messagesString;
	if(this->params.newestMessageAtBottom){
		for(int i = 0; i < this->messagesIndex+1; i++){
		messagesString = messagesString + (i == messagesIndex ? "<b>"+this->messages.at(i)+"</b>" : this->messages.at(i));
		}
	} else {
		//append all messages to messagesString and make first one bold
		for(int i = this->messagesIndex+MAX_MESSAGES; i > this->messagesIndex; i--){
		messagesString = messagesString + (i == messagesIndex+MAX_MESSAGES ? "<b>"+this->messages.at(i%MAX_MESSAGES)+"</b>" : this->messages.at(i%MAX_MESSAGES));
		}
	}
	this->textEdit->setText(messagesString);

	if(this->messagesIndex>0){
		this->messagesIndex++;
	}

	if(this->params.newestMessageAtBottom){
		this->textEdit->moveCursor(QTextCursor::End);
		this->textEdit->ensureCursorVisible();
	}
}

void MessageConsole::resizeEvent(QResizeEvent *event) {
//this code saves the console size in params to restore it on app restart.
//however, due to a bug in Qt 5.11, the MessageConsole gets resized without user action when other docks are modified.
//see Qt bug report:https://bugreports.qt.io/browse/QTBUG-65592.
//this bug has been fixed for qt 5.12, see here: https://code.qt.io/cgit/qt/qtbase.git/commit/src/widgets/widgets/qdockarealayout.cpp?id=e2d79b496335e8d8666014e900930c66cf722eb6
//the following code serves as a workaround for Qt version older than 5.12.
#if QT_VERSION < QT_VERSION_CHECK(5, 12, 0)
	const int heightChangeThreshold = 50;
	if (qAbs(event->oldSize().height() - event->size().height()) < heightChangeThreshold) {
		this->params.preferredHeight = this->height();
	} else {
		int minHeight = this->minimumHeight();
		this->setFixedHeight(this->params.preferredHeight);
		this->setMinimumHeight(minHeight);
		this->setMaximumHeight(QWIDGETSIZE_MAX);
	}
#else
	this->params.preferredHeight = this->height();
#endif
	QWidget::resizeEvent(event);
}

void MessageConsole::insertNewMessagesAtBottom(bool enable) {
	if(this->params.newestMessageAtBottom == enable){
		return;
	}
	this->params.newestMessageAtBottom = enable;
	this->refreshMessages();
}

void MessageConsole::displayInfo(QString info) {
	QString currentTime = QDateTime::currentDateTime().toString("hh:mm:ss") + " ";
	QString htmlStart = "<font color=\"#4863A0\">";	//#4863A0 = "Steel Blue", RGB: 72, 99, 160, (http://www.computerhope.com/htmcolor.htm#color-codes)
	QString htmlEnd = "</font>";
	this->textEdit->setText(addStringToMessageBuffer(currentTime + htmlStart + info + htmlEnd));

	if(this->params.newestMessageAtBottom){
		this->textEdit->moveCursor(QTextCursor::End);
		this->textEdit->ensureCursorVisible();
	}
}

void MessageConsole::displayError(QString error) {
	QString currentTime = QDateTime::currentDateTime().toString("hh:mm:ss") + " ";
	QString htmlStart = "<font color=\"#E41B17\">";	//#E41B17 = "Love Red" (http://www.computerhope.com/htmcolor.htm#color-codes)
	QString htmlEnd = "</font>";

	//todo: maybe don't use a QVector<QString> as message buffer to generate a string for textEdit. instead maybe add new messages like this:
	//	auto cursor = QTextCursor(this->textEdit->document());
	//	cursor.setPosition(0);
	//	this->textEdit->setTextCursor(cursor);
	//	this->textEdit->insertHtml((currentTime + htmlStart + error + htmlEnd + "<br>"));

	this->textEdit->setText(addStringToMessageBuffer(currentTime + htmlStart + "<b>ERROR: </b>" + error + htmlEnd));

	if(this->params.newestMessageAtBottom){
		this->textEdit->moveCursor(QTextCursor::End);
		this->textEdit->ensureCursorVisible();
	}
}
