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

#include "aboutdialog.h"
#include <QVBoxLayout>
#include <QTabWidget>
#include <QTextEdit>
#include <QFile>
#include <QTextStream>
#include <QTextBrowser>
#include <QPushButton>
#include <QCoreApplication>
#include <QLabel>
#include <QDebug>

AboutDialog::AboutDialog(QWidget *parent) : QDialog(parent)
{
	//window settings
	this->setWindowTitle(tr("About OCTproZ"));
	this->setWindowOpacity(0.90);
	this->setMinimumWidth(768);
	this->setMinimumHeight(256);

	//setup left area of about dialog with logo
	QVBoxLayout *vLayoutLeft = new QVBoxLayout();
	vLayoutLeft->setSpacing(0);
	QLabel* labelWithLogo = new QLabel(this);
	QPixmap pix(":/aboutdata/octproz_logo.png");
	labelWithLogo->setPixmap(pix);
	vLayoutLeft->addWidget(labelWithLogo, 0, Qt::AlignHCenter);
	QLabel* labelWithAuthorInfo = new QLabel(this);
	labelWithAuthorInfo->setText("Author: Miroslav Zabic\n" \
				  "Contact: zabic" \
				  "@" \
				  "iqo.uni-hannover.de\n\n" \
				  "Version: " + qApp->applicationVersion());
	labelWithAuthorInfo->setTextInteractionFlags(Qt::TextSelectableByMouse | Qt::TextSelectableByKeyboard);
	vLayoutLeft->addWidget(labelWithAuthorInfo);

	//setup right area of about dialog with tabwidget
	QHBoxLayout* hLayoutTop = new QHBoxLayout();
	hLayoutTop->addLayout(vLayoutLeft);
	vLayoutLeft->setContentsMargins(2, 15, 2, 0);
	QTabWidget *tabWidget = new QTabWidget(this);
	hLayoutTop->addWidget(tabWidget);

	//about
	QString aboutText = tr("<b>OCTproZ</b> is an open source software for online processig of optical coherence tomography (OCT) raw data. "
			  "It can be extended by plugins, which are divided into two kinds: systems and extensions. Systems are software "
			  "representations of actual OCT systems and provide raw data. Extensions are software modules that extend the "
			  "functionality of an OCT system and/or OCTproZ.");
	QTextEdit* aboutTextEdit = new QTextEdit(this);
	aboutTextEdit->setReadOnly(true);
	aboutTextEdit->setText(aboutText);
	tabWidget->addTab(aboutTextEdit, tr("About"));

	//license
	QString licenseText = ("OCTproZ is free software: you can redistribute it and/or modify "
						  "it under the terms of the GNU General Public License as published by "
						  "the Free Software Foundation, either version 3 of the License, or "
						  "(at your option) any later version.<br>"
						  "This program is distributed in the hope that it will be useful, "
						  "but WITHOUT ANY WARRANTY; without even the implied warranty of "
						  "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the "
						  "GNU General Public License for more details.<br><br><hr><pre>");

	QFile licenseFile(":/aboutdata/LICENSE");
	Q_ASSERT(licenseFile.exists());
	licenseFile.open(QIODevice::ReadOnly);
	QByteArray licenseByteArray = licenseFile.readAll();
	licenseText.append(QString::fromUtf8(licenseByteArray));
	licenseText.append("</pre></body></html>");
	QTextEdit* licenceTextEdit = new QTextEdit(this);
	licenceTextEdit->setReadOnly(true);
	licenceTextEdit->setText(licenseText);
	tabWidget->addTab(licenceTextEdit, tr("License"));

	//credits
	QString creditsText = "";
	QTextEdit* creditsTextEdit = new QTextEdit(this);
	creditsTextEdit->setReadOnly(true);

	QFile creditsFile(":/aboutdata/credits.txt");
	Q_ASSERT(creditsFile.exists());
	creditsFile.open(QIODevice::ReadOnly);
	QByteArray creditsByteArray = creditsFile.readAll();
	creditsText.append(QString::fromUtf8(creditsByteArray));
	creditsTextEdit->setText(creditsText);
	tabWidget->addTab(creditsTextEdit, tr("Credits"));

	//third-party software components
	QString thirdpartyText = tr("<html><body><h2>Third party components used by OCTproZ:</h2><ul>");
	QTextEdit* thirdpartyTextEdit = new QTextEdit(this);
	thirdpartyTextEdit->setReadOnly(true);
	thirdpartyTextEdit->setTextInteractionFlags(Qt::LinksAccessibleByMouse | Qt::LinksAccessibleByKeyboard);

	QFile thirdpartyFile(":/aboutdata/thirdparty.txt");
	Q_ASSERT(thirdpartyFile.exists());
	thirdpartyFile.open(QIODevice::ReadOnly);
	QByteArray thirdpartyByteArray = thirdpartyFile.readAll();
	thirdpartyText.append(QString::fromUtf8(thirdpartyByteArray));
	thirdpartyText.append("</ul></body></html>");
	thirdpartyTextEdit->setText(thirdpartyText);
	tabWidget->addTab(thirdpartyTextEdit, tr("Third-party components"));

	//close buttom
	QPushButton *closeButton = new QPushButton(tr("Close"));
	connect(closeButton, &QPushButton::clicked, this, &AboutDialog::close);
	QHBoxLayout *hLayoutBottom = new QHBoxLayout;
	hLayoutBottom->setMargin(6);
	hLayoutBottom->addStretch(10);
	hLayoutBottom->addWidget(closeButton);

	//main layout
	QVBoxLayout *vLayoutMain = new QVBoxLayout(this);
	int defaultMargin = vLayoutMain->margin() +10;
	int defaultSpacing = vLayoutMain->spacing();
	vLayoutMain->setSpacing(defaultSpacing+10);
	vLayoutMain->setContentsMargins(defaultMargin, defaultMargin, defaultMargin, 0);
	vLayoutMain->addLayout(hLayoutTop);
	vLayoutMain->addLayout(hLayoutBottom);

	connect(tabWidget, &QTabWidget::tabBarDoubleClicked, this, &AboutDialog::easterEgg);
}

AboutDialog::~AboutDialog() {
}
