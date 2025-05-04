/**
**  This file is part of OCTproZ.
**  OCTproZ is an open source software for processig of optical
**  coherence tomography (OCT) raw data.
**  Copyright (C) 2019-2025 Miroslav Zabic
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
#include <QSplitter>
#include <QListWidget>

AboutDialog::AboutDialog(QWidget *parent) : QDialog(parent)
{
	setupWindowProperties();

	QVBoxLayout *vLayoutLeft = createLogoLayout();
	QTabWidget *tabWidget = new QTabWidget(this);

	QHBoxLayout* hLayoutTop = new QHBoxLayout();
	hLayoutTop->addLayout(vLayoutLeft);
	hLayoutTop->addWidget(tabWidget);

	setupTabs(tabWidget);

	QHBoxLayout *hLayoutBottom = createButtonLayout();

	QVBoxLayout *vLayoutMain = new QVBoxLayout(this);
	int defaultMargin = vLayoutMain->margin() + 10;
	int defaultSpacing = vLayoutMain->spacing();
	vLayoutMain->setSpacing(defaultSpacing + 10);
	vLayoutMain->setContentsMargins(defaultMargin, defaultMargin, defaultMargin, 0);
	vLayoutMain->addLayout(hLayoutTop);
	vLayoutMain->addLayout(hLayoutBottom);

	setupConnections(tabWidget);
}

void AboutDialog::setupWindowProperties() {
	setWindowTitle(tr("About OCTproZ"));
	setWindowOpacity(0.90);
	setMinimumWidth(768);
	setMinimumHeight(256);
}

QVBoxLayout* AboutDialog::createLogoLayout() {
	QVBoxLayout *vLayoutLeft = new QVBoxLayout();
	vLayoutLeft->setSpacing(0);
	vLayoutLeft->setContentsMargins(2, 15, 2, 0);

	QLabel* labelWithLogo = new QLabel(this);
	QPixmap pix(":/aboutdata/octproz_logo.png");
	labelWithLogo->setPixmap(pix);
	vLayoutLeft->addWidget(labelWithLogo, 0, Qt::AlignHCenter);

	QLabel* labelWithAuthorInfo = new QLabel(this);
	labelWithAuthorInfo->setText("Author: Miroslav Zabic\n" \
				  "Contact: zabic" \
				  "@" \
				  "spectralcode.de\n\n" \
				  "Version: " + qApp->applicationVersion());
	labelWithAuthorInfo->setTextInteractionFlags(Qt::TextSelectableByMouse | Qt::TextSelectableByKeyboard);
	vLayoutLeft->addWidget(labelWithAuthorInfo);

	labelWithLogo->setContextMenuPolicy(Qt::ContextMenuPolicy::CustomContextMenu);
	connect(labelWithLogo, &QLabel::customContextMenuRequested, this, &AboutDialog::easterEgg);

	return vLayoutLeft;
}

void AboutDialog::setupTabs(QTabWidget *tabWidget) {
	addAboutTab(tabWidget);
	addLicenseTab(tabWidget);
	addCreditsTab(tabWidget);
	addThirdPartyTab(tabWidget);
}

void AboutDialog::addAboutTab(QTabWidget *tabWidget) {
	QString aboutText;
	QFile aboutFile(":/aboutdata/about.txt");

	if (aboutFile.exists() && aboutFile.open(QIODevice::ReadOnly)) {
		QByteArray aboutByteArray = aboutFile.readAll();
		aboutText = QString::fromUtf8(aboutByteArray);
		aboutFile.close();
	} else {
		aboutText = "About file could not be opened.";
		qWarning() << "Failed to open about file";
	}

	QTextEdit* aboutTextEdit = new QTextEdit(this);
	aboutTextEdit->setReadOnly(true);
	aboutTextEdit->setText(aboutText);
	tabWidget->addTab(aboutTextEdit, tr("About"));
}

void AboutDialog::addLicenseTab(QTabWidget *tabWidget) {
	QString licenseText = ("OCTproZ is free software: you can redistribute it and/or modify "
						  "it under the terms of the GNU General Public License as published by "
						  "the Free Software Foundation, either version 3 of the License, or "
						  "(at your option) any later version.<br>"
						  "This program is distributed in the hope that it will be useful, "
						  "but WITHOUT ANY WARRANTY; without even the implied warranty of "
						  "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the "
						  "GNU General Public License for more details.<br><br><hr><pre>");

	QFile licenseFile(":/LICENSE");
	if (licenseFile.exists() && licenseFile.open(QIODevice::ReadOnly)) {
		QByteArray licenseByteArray = licenseFile.readAll();
		licenseText.append(QString::fromUtf8(licenseByteArray));
		licenseFile.close();
	} else {
		licenseText.append("License file could not be opened.");
		qWarning() << "Failed to open license file";
	}
	licenseText.append("</pre></body></html>");

	QTextEdit* licenceTextEdit = new QTextEdit(this);
	licenceTextEdit->setReadOnly(true);
	licenceTextEdit->setText(licenseText);
	tabWidget->addTab(licenceTextEdit, tr("License"));
}

void AboutDialog::addCreditsTab(QTabWidget *tabWidget) {
	QString creditsText;
	QFile creditsFile(":/contributors.txt");

	if (creditsFile.exists() && creditsFile.open(QIODevice::ReadOnly)) {
		QByteArray creditsByteArray = creditsFile.readAll();
		creditsText = QString::fromUtf8(creditsByteArray);
		creditsFile.close();
	} else {
		creditsText = "Credits file could not be opened.";
		qWarning() << "Failed to open credits file";
	}

	QTextEdit* creditsTextEdit = new QTextEdit(this);
	creditsTextEdit->setReadOnly(true);
	creditsTextEdit->setText(creditsText);
	tabWidget->addTab(creditsTextEdit, tr("Credits"));
}

void AboutDialog::addThirdPartyTab(QTabWidget *tabWidget) {
	QWidget* thirdPartyWidget = new QWidget(this);
	QHBoxLayout* thirdPartyLayout = new QHBoxLayout(thirdPartyWidget);
	thirdPartyLayout->setContentsMargins(0, 0, 0, 0);

	QSplitter* splitter = new QSplitter(Qt::Horizontal, thirdPartyWidget);
	thirdPartyLayout->addWidget(splitter);

	QListWidget* componentList = new QListWidget(splitter);
	componentList->setMinimumWidth(100);
	componentList->setMaximumWidth(180);

	QTextBrowser* componentDetails = new QTextBrowser(splitter);
	componentDetails->setOpenExternalLinks(true);

	splitter->addWidget(componentList);
	splitter->addWidget(componentDetails);
	splitter->setStretchFactor(0, 1);
	splitter->setStretchFactor(1, 4);

	QList<int> sizes;
	sizes << 100 << 600;
	splitter->setSizes(sizes);

	QList<ComponentInfo> components = getThirdPartyComponents();

	for (int i = 0; i < components.size(); ++i) {
		componentList->addItem(components.at(i).name);
	}

	connect(componentList, &QListWidget::currentTextChanged, this, [this, componentDetails, components](const QString& componentName) {
			for (int i = 0; i < components.size(); ++i) {
				const ComponentInfo& component = components.at(i);
				if (component.name == componentName) {
					updateComponentDetails(componentDetails, component);
					break;
				}
			}
		}
	);

	if (componentList->count() > 0) {
		componentList->setCurrentRow(0);
	}

	tabWidget->addTab(thirdPartyWidget, tr("Third-party components"));
}

QList<ComponentInfo> AboutDialog::getThirdPartyComponents() {
	QList<ComponentInfo> components;

	ComponentInfo cuda;
	cuda.name = "Cuda";
	cuda.url = "https://developer.nvidia.com/cuda-toolkit";
	cuda.licensePath = ":/aboutdata/thirdparty_licenses/cuda_license.txt";
	components.append(cuda);

	ComponentInfo eigen;
	eigen.name = "Eigen";
	eigen.url = "https://eigen.tuxfamily.org";
	eigen.licensePath = ":/aboutdata/thirdparty_licenses/eigen_license.txt";
	components.append(eigen);

	ComponentInfo fftw;
	fftw.name = "FFTW";
	fftw.url = "http://www.fftw.org";
	fftw.licensePath = ":/aboutdata/thirdparty_licenses/fftw_license.txt";
	components.append(fftw);

	ComponentInfo qcustomplot;
	qcustomplot.name = "QCustomPlot";
	qcustomplot.url = "http://www.qcustomplot.com/ ";
	qcustomplot.licensePath = ":/aboutdata/thirdparty_licenses/qcustomplot_license.txt";
	components.append(qcustomplot);

	ComponentInfo qt;
	qt.name = "Qt Framework";
	qt.url = "https://www.qt.io";
	qt.licensePath = ":/aboutdata/thirdparty_licenses/qt_license.txt";
	components.append(qt);

	ComponentInfo raycaster;
	raycaster.name = "Raycaster";
	raycaster.url = "https://github.com/m-pilia/volume-raycasting";
	raycaster.licensePath = ":/aboutdata/thirdparty_licenses/raycaster_license.txt";
	components.append(raycaster);

	return components;
}

QString AboutDialog::loadLicenseText(const QString& path) {
	QString licenseText;
	QFile licenseFile(path);

	if (licenseFile.exists() && licenseFile.open(QIODevice::ReadOnly)) {
		licenseText = QString::fromUtf8(licenseFile.readAll());
		licenseFile.close();
	} else {
		licenseText = tr("License file could not be opened: %1").arg(path);
		qWarning() << "Failed to open license file:" << path;
	}

	return licenseText;
}

void AboutDialog::updateComponentDetails(QTextBrowser* browser, const ComponentInfo& component) {
	QString detailsText = "<h2>" + component.name + "</h2>";

	if (!component.url.isEmpty()) {
		detailsText += "<p><b>" + tr("Homepage") + ":</b> <a href='" +
					component.url + "'>" + component.url + "</a></p>";
	}

	detailsText += "<h3>" + tr("License") + ":</h3><pre>";
	detailsText += loadLicenseText(component.licensePath);
	detailsText += "</pre>";

	browser->setHtml(detailsText);
}

QHBoxLayout* AboutDialog::createButtonLayout() {
	QPushButton *closeButton = new QPushButton(tr("Close"));
	connect(closeButton, &QPushButton::clicked, this, &AboutDialog::close);

	QHBoxLayout *hLayoutBottom = new QHBoxLayout;
	hLayoutBottom->setMargin(6);
	hLayoutBottom->addStretch(10);
	hLayoutBottom->addWidget(closeButton);

	return hLayoutBottom;
}

void AboutDialog::setupConnections(QTabWidget *tabWidget) {
	connect(tabWidget, &QTabWidget::tabBarDoubleClicked, this, &AboutDialog::easterEgg);
}

AboutDialog::~AboutDialog() {
}
