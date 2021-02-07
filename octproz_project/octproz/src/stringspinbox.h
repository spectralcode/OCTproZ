/**
**  This file is part of OCTproZ.
**  OCTproZ is an open source software for processig of optical
**  coherence tomography (OCT) raw data.
**  Copyright (C) 2019-2021 Miroslav Zabic
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

#ifndef STRINGSPINBOX_H
#define STRINGSPINBOX_H


#include <QAbstractSpinBox>
#include <QLineEdit>



class StringSpinBox : public QAbstractSpinBox
{
	Q_OBJECT

public:
	StringSpinBox(QWidget* parent = nullptr);
	~StringSpinBox();

	void setStrings(QStringList strings);

	virtual void stepBy(int steps) override;
	int getIndex(){return this->currentIndex;}
	void setIndex(int index);
	QString getText(){return this->strings.size() == 0 ? "" : this->strings.at(this->currentIndex);}


private:
	void adjustWidth();

	int currentIndex;
	QStringList strings;

protected:
	virtual StepEnabled stepEnabled() const override;

signals:
	void indexChanged();
};



#endif  // STRINGSPINBOX_H
