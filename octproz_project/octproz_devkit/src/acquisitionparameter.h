/*
MIT License

Copyright (c) 2019-2021 Miroslav Zabic

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

#ifndef ACQUISITIONPARAMETER_H
#define ACQUISITIONPARAMETER_H


#include <QObject>

struct AcquisitionParams {
	unsigned int samplesPerLine;
	unsigned int ascansPerBscan;
	unsigned int bscansPerBuffer;
	unsigned int buffersPerVolume;
	unsigned int bitDepth;
};

class AcquisitionParameter : public QObject
{
	Q_OBJECT

public:
	AcquisitionParameter(QObject *parent = nullptr);
	~AcquisitionParameter();

	//params are public just for testing //todo: params should be private. add private getter methods for params
	AcquisitionParams params;

private:
	

public slots :
	void slot_updateParams(AcquisitionParams newParams);

signals:
	void updated(AcquisitionParams newParams);

};

#endif // ACQUISITIONPARAMETER_H
