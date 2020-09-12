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

#ifndef ACQUISITIONBUFFER_H
#define ACQUISITIONBUFFER_H

#include <qobject.h>
#include <qvector.h>
#include <qstring.h>

#ifdef _WIN32
	#include <conio.h>
	#include <Windows.h>
	#define posix_memalign(p, a, s) (((*(p)) = _aligned_malloc((s), (a))), *(p) ? 0 : errno)
	#define posix_memalign_free _aligned_free
#else
	#include <stdlib.h>
	#define posix_memalign_free free
#endif


class AcquisitionBuffer : public QObject
{
	Q_OBJECT
public:
	AcquisitionBuffer();
	~AcquisitionBuffer();

	bool allocateMemory(unsigned int bufferCnt, size_t bytesPerBuffer);
	void releaseMemory();

	QVector<void*> bufferArray;
	QVector<bool> bufferReadyArray;
	int currIndex;
	unsigned int bufferCnt;
	size_t bytesPerBuffer;

private:


public slots:


signals:
	void error(QString);
	void info(QString);
};

#endif // ACQUISITIONBUFFER_H
