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

#ifndef ACQUISITIONSYSTEM_H
#define ACQUISITIONSYSTEM_H

#include <QObject>
#include <QEventLoop>
#include <QDialog>
#include <QDebug>
#include <QThread>
#include "acquisitionbuffer.h"
#include "acquisitionparameter.h"
#include "plugin.h"


class AcquisitionSystem : public Plugin
{
	Q_OBJECT
public:
	//! Constructor
	/*!
	 * Constructor is called automatically during startup of OCTproZ.
	 * Type, name and tooltip should be set in the constructor.
	*/
	AcquisitionSystem();

	//! Destructor
	/*!
	 * Destructor is called automatically when OCTproZ is exited.
	*/
	~AcquisitionSystem();

	/*!
	 * \brief startAcquisition is a pure virtual function for initializing and starting the acquisition process. It is called by OCTproZ after the user has pressed the "start" button.
	 */
	virtual void startAcquisition() = 0;

	/*!
	 * \brief stopAcquisition is a pure virtual function for uninitializing and stopping the acquisition process. It is called by OCTproZ after the user has pressed the "stop" button or on closing of OCTproZ.
	 */
	virtual void stopAcquisition() = 0;

	AcquisitionBuffer* buffer; ///< Page aligned memory buffer for acquisition data
	AcquisitionParameter* params; ///< Acquisition parameters: bit depth, samples per line, lines per frame (ascansPerBscan), frames per buffer (bscansPerBuffer), buffers per volume
	QDialog* settingsDialog; ///< Dialog that is displayed to the user
	bool acqusitionRunning; ///< This bool must be set to true when the acquisition is running, otherwise it must be set to false. It is used by OCTproZ to determine if acquisitioin is running.


signals:
	void acquisitionStarted(AcquisitionSystem*); ///< This signal must be emitted when the actual acquisition starts. It signalsOCTproZ to start processing.
	void acquisitionStopped(); ///< This signal must be emitted when the actual acquisition stops.

};

#define AcquisitionSystem_iid "octproz.acquisition.interface"

Q_DECLARE_INTERFACE(AcquisitionSystem, AcquisitionSystem_iid)

#endif // ACQUISITIONSYSTEM_H

