/*
MIT License

Copyright (c) 2019-2020 Miroslav Zabic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "demoextensionform.h"
#include "ui_demoextensionform.h"

DemoExtensionForm::DemoExtensionForm(QWidget *parent) :
	QWidget(parent),
	ui(new Ui::DemoExtensionForm)
{
	this->ui->setupUi(this);

	//Connect gui elements to update parameters every time user changes them
	connect(this->ui->checkBox_like, &QCheckBox::clicked, this, &DemoExtensionForm::apply);
	connect(this->ui->doubleSpinBox_demoParameter, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &DemoExtensionForm::apply);
}

DemoExtensionForm::~DemoExtensionForm()
{
	delete this->ui;
}

void DemoExtensionForm::setSettings(QVariantMap settings){
	this->ui->checkBox_like->setChecked(settings.value(LIKE).toBool());
	this->ui->doubleSpinBox_demoParameter->setValue(settings.value(PARAM).toDouble());
	this->apply();
}

void DemoExtensionForm::getSettings(QVariantMap* settings) {
	settings->insert(LIKE, this->parameters.like);
	settings->insert(PARAM, this->parameters.demoParam);
}

void DemoExtensionForm::apply() {
	this->parameters.like = this->ui->checkBox_like->isChecked();
	this->parameters.demoParam = this->ui->doubleSpinBox_demoParameter->value();
	emit parametersUpdated(this->parameters);
}
