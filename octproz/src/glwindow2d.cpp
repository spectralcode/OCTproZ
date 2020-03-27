/**
**  This file is part of OCTproZ.
**  OCTproZ is an open source software for processig of optical
**  coherence tomography (OCT) raw data.
**  Copyright (C) 2019-2020 OCTproZ developer
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

#include "glwindow2d.h"


GLWindow2D::GLWindow2D(QWidget *parent) : QOpenGLWidget(parent){
	this->width = DEFAULT_WIDTH;
	this->height = DEFAULT_HEIGHT;
	this->depth = 0;
	this->scaleFactor = 1.0;
	this->screenWidthScaled = 1.0;
	this->screenHeightScaled = 1.0;
	this->xTranslation = 0.0;
	this->yTranslation = 0.0;
	this->setMinimumWidth(448);
	this->setMinimumHeight(448);
	this->initialized = false;
	this->changeBufferSizeFlag = false;
	this->keepAspectRatio = true;
	this->frameNr = 0;
	this->rotationAngle = 0.0;
	this->markerVisible = false;
	this->setMarkerOrigin(LEFT);
	this->slot_setMarkerPosition(0);
	this->setFocusPolicy(Qt::StrongFocus);
	this->panel = new ControlPanel2D(this);
	this->layout = new QVBoxLayout(this);
	this->layout->addStretch();
	this->layout->addWidget(this->panel);
	this->panel->setVisible(false);

	this->initContextMenu();

	connect(this->panel->doubleSpinBoxRotationAngle, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &GLWindow2D::setRotationAngle);
	connect(this->panel->spinBoxFrame, QOverload<int>::of(&QSpinBox::valueChanged), this, &GLWindow2D::currentFrameNr);
}


GLWindow2D::~GLWindow2D()
{
	//todo: check if cleanup (probably for processingContext and processingSurface) is necessary and implement it
}

void GLWindow2D::setMarkerOrigin(FRAME_EDGE markerOrigin) {
	this->markerOrigin = markerOrigin;
}

void GLWindow2D::initContextMenu(){
	this->contextMenu = new QMenu(this);

	this->keepAspectRatioAction = new QAction(tr("Keep &Aspect Ratio"), this);
	this->keepAspectRatioAction->setCheckable(true);
	this->keepAspectRatioAction->setChecked(this->keepAspectRatio);
	connect(this->keepAspectRatioAction, &QAction::toggled, this, &GLWindow2D::setKeepAspectRatio);
	this->contextMenu->addAction(this->keepAspectRatioAction);

	this->markerAction = new QAction(tr("Display orthogonal &marker"), this);
	this->markerAction->setCheckable(true);
	this->markerAction->setChecked(this->markerVisible);
	connect(this->markerAction, &QAction::toggled, this, &GLWindow2D::slot_displayMarker);
	contextMenu->addAction(this->markerAction);

	this->screenshotAction = new QAction(tr("&Screenshot..."), this);
	connect(this->screenshotAction, &QAction::triggered, this, &GLWindow2D::slot_screenshot);
	this->contextMenu->addAction(this->screenshotAction);
}

void GLWindow2D::slot_changeBufferAndTextureSize(unsigned int width, unsigned int height, unsigned int depth) {
	this->width = width <= 1 ? 128 : width; //width and height shall not be zero. 128 is an arbitrary value greater 1 and a power of 2
	this->height = height <= 1 ? 128 : height;
	this->depth = depth;
	this->panel->setMaxFrame(depth-1);
	this->panel->setMaxAverage(depth-1);

	if(!this->initialized){
		this->changeBufferSizeFlag = true;
		return;
	}
	this->resizeGL((float)this->size().width(),(float)this->size().height());

	makeCurrent();
	glDeleteBuffers(1, &(this->buf));
	glGenBuffers(1, &(this->buf));
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, this->buf);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, (this->width * this->height * sizeof(float)), 0, GL_DYNAMIC_COPY);
	glBindTexture(GL_TEXTURE_2D, this->texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, this->width, this->height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
	doneCurrent();
	emit registerBufferCudaGL(this->buf);

	this->slot_setMarkerPosition(this->markerPosition);
}

void GLWindow2D::slot_initProcessingThreadOpenGL(QOpenGLContext* processingContext, QOffscreenSurface* processingSurface, QThread* processingThread) {
	QOpenGLContext* renderContext = this->context();
	(processingContext)->setFormat(renderContext->format());
	(processingContext)->setShareContext(renderContext);
	(processingContext)->create();
	(processingContext)->moveToThread(processingThread);
	(processingSurface)->setFormat(renderContext->format());
	(processingSurface)->create(); //Due to the fact that QOffscreenSurface is backed by a QWindow on some platforms, cross-platform applications must ensure that create() is only called on the main (GUI) thread
	(processingSurface)->moveToThread(processingThread);
	
	this->slot_changeBufferAndTextureSize(this->width, this->height, this->depth);

	//QOpenGLContext::areSharing(processingContext, renderContext) ? emit info("processingContext, renderContext: yes") : emit info("processingContext, renderContext: no");
}

void GLWindow2D::slot_registerGLbufferWithCuda() {
	emit registerBufferCudaGL(this->buf);
}

void GLWindow2D::setKeepAspectRatio(bool keepAspectRatio) {
	this->keepAspectRatio = keepAspectRatio;
	this->resizeGL((float)this->size().width(),(float)this->size().height());
}

void GLWindow2D::setRotationAngle(double rotationAngle) {
    this->rotationAngle = rotationAngle;
    this->resizeGL((float)this->size().width(),(float)this->size().height());
}

void GLWindow2D::slot_screenshot() {
	QImage screenshot = this->grabFramebuffer();
	emit dialogAboutToOpen();
	QCoreApplication::processEvents();
	QString filters("(*.png);;(*.jpg);;(*.bmp)");
	QString defaultFilter("(*.png)");
	QString fileName = QFileDialog::getSaveFileName(this, tr("Save screenshot to..."), QDir::currentPath(), filters, &defaultFilter);
	emit dialogClosed();
	if(fileName == ""){
		return;
	}
	bool saved = false;
	saved = screenshot.save(fileName);

	if(saved){
		emit info(tr("Screenshot saved to ") + fileName);
	}else{
		emit error(tr("Could not save screenshot to disk."));
	}
}

void GLWindow2D::slot_displayMarker(bool display) {
	this->markerVisible = display;
}

void GLWindow2D::slot_setMarkerPosition(unsigned int position) {
	this->markerPosition = position;
	float linePos = 0;
	float markerTexturePosition = 0;

	switch(this->markerOrigin){
	case TOP:
		linePos = (2.0f*this->markerPosition)/static_cast<float>(this->width)-1.0f;
		markerTexturePosition = this->screenHeightScaled*linePos;
		this->markerCoordinates.x1 = this->screenWidthScaled;
		this->markerCoordinates.x2 = -this->screenWidthScaled;
		this->markerCoordinates.y1 = -markerTexturePosition;
		this->markerCoordinates.y2 = -markerTexturePosition;
		break;
	case BOTTOM:
		linePos = (2.0f*this->markerPosition)/static_cast<float>(this->width)-1.0f;
		markerTexturePosition = this->screenHeightScaled*linePos;
		this->markerCoordinates.x1 = this->screenWidthScaled;
		this->markerCoordinates.x2 = -this->screenWidthScaled;
		this->markerCoordinates.y1 = markerTexturePosition;
		this->markerCoordinates.y2 = markerTexturePosition;
		break;
	case LEFT:
		linePos = (2.0f*this->markerPosition)/static_cast<float>(this->height)-1.0f;
		markerTexturePosition = this->screenWidthScaled*linePos;
		this->markerCoordinates.x1 = markerTexturePosition;
		this->markerCoordinates.x2 = markerTexturePosition;
		this->markerCoordinates.y1 = this->screenHeightScaled;
		this->markerCoordinates.y2 = -this->screenHeightScaled;
		break;
	case RIGHT:
		linePos = (2.0f*this->markerPosition)/static_cast<float>(this->height)-1.0f;
		markerTexturePosition = this->screenWidthScaled*linePos;
		this->markerCoordinates.x1 = -markerTexturePosition;
		this->markerCoordinates.x2 = -markerTexturePosition;
		this->markerCoordinates.y1 = this->screenHeightScaled;
		this->markerCoordinates.y2 = -this->screenHeightScaled;
		break;
	}
}

void GLWindow2D::slot_saveScreenshot(QString savePath, QString fileName) {
	QImage screenshot = this->grabFramebuffer();
	QString filePath = savePath + "/" + fileName;
	screenshot.save(filePath);
	emit info(tr("Screenshot saved to ") + filePath);
}

void GLWindow2D::initializeGL(){
	initializeOpenGLFunctions();

	//check if width and height are greater 1. if not, set them to the (arbitrary) value 128
	if(this->width <= 1){
		this->width = 128;
	}
	if(this->height <= 1){
		this->height = 128;
	}

	glGenBuffers(1, &buf);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, buf);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, (this->width * this->height * sizeof(float)), 0, GL_DYNAMIC_COPY);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glGenTextures(1, &(this->texture));
	glBindTexture(GL_TEXTURE_2D, this->texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, this->width, this->height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	if(this->initialized){
		emit registerBufferCudaGL(this->buf); //registerBufferCudaGL is necessary here because as soon as the openglwidget/dock is removed from the main window initializeGL() is called again.
	}

	this->initialized = true;

	if(this->changeBufferSizeFlag){
		this->slot_changeBufferAndTextureSize(this->width, this->height, this->depth);
		this->changeBufferSizeFlag = false;
	}
}

void GLWindow2D::paintGL(){
    glLoadIdentity();
    glTranslatef(this->xTranslation, this->yTranslation, 0); //verschieben
//    glRotatef(this->rotationAngle, 0.0, 0.0, 1.0);
    glScalef(this->scaleFactor, this->scaleFactor, 0.f); //zoom
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, this->buf);
    glBindTexture(GL_TEXTURE_2D, this->texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->width, this->height, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glEnable(GL_TEXTURE_2D);
    glColor3f(1.0, 1.0, 1.0); //Backgroundcolor
    glBegin(GL_QUADS);
        glTexCoord2f(0, 1), glVertex2f(this->screenWidthScaled01,this->screenHeightScaled01); //place upper left texture coordinate to bottom left screen position
        glTexCoord2f(1, 1), glVertex2f(this->screenWidthScaled11, this->screenHeightScaled11); //upper right texture coordinate to upper left screen position
        glTexCoord2f(1, 0), glVertex2f(this->screenWidthScaled10,this->screenHeightScaled10); //bottom right texture coordinate to upper right screen position
        glTexCoord2f(0, 0), glVertex2f(this->screenWidthScaled00, this->screenHeightScaled00); //bottom left texture coordinate to bottom right screen position
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);


	//dispaly marker
	if(this->markerVisible){
		glLineWidth(3.0);
		glBegin(GL_LINES);
			glColor3f(0.8f, 0.0f, 0.0f);
			glVertex3f(this->markerCoordinates.x1, this->markerCoordinates.y1, 0.0);
			glVertex3f(this->markerCoordinates.x2, this->markerCoordinates.y2, 0.0);
		glEnd();
	}

	//request next frame
	update();
}

//todo: add rotational angle calculation for screenheightScaled und screenWidthScaled!
void GLWindow2D::resizeGL(int w, int h){
    double deg_to_rad = 3.1415926535898/180.0;
    double rotationAngle_rad = rotationAngle*deg_to_rad;

    if(this->keepAspectRatio){
        float textureWidth = static_cast<float>(this->height);
        float textureHeight = static_cast<float>(this->width);
        float screenWidth = static_cast<float>(w);//(float)this->size().width();
        float screenHeight = static_cast<float>(h);//(float)this->size().height();
        float ratioTexture = textureWidth/textureHeight;
        float ratioScreen = screenWidth/screenHeight;
        this->screenWidthScaled = 1;
        this->screenHeightScaled = 1;

        if(ratioTexture > ratioScreen){
            this->screenHeightScaled = ratioScreen/ratioTexture;
        }
        else{
            this->screenWidthScaled =  ratioTexture/ratioScreen;           
        }
        this->screenWidthScaled = this->screenWidthScaled*screenWidth/screenHeight; //transformation to uniformly x-axis
        //vertex01 bottom left
        this->screenWidthScaled01 = -this->screenWidthScaled*cos(rotationAngle_rad)+this->screenHeightScaled*sin(rotationAngle_rad); //rotaion x-coordinate
        this->screenHeightScaled01 = -this->screenWidthScaled*sin(rotationAngle_rad)-this->screenHeightScaled*cos(rotationAngle_rad); //rotation y-coordinate
        this->screenWidthScaled01 = screenWidthScaled01*screenHeight/screenWidth; //back-transformation to streched x-axis
        //vertex11 top left
        this->screenWidthScaled11 = -this->screenWidthScaled*cos(rotationAngle_rad)-this->screenHeightScaled*sin(rotationAngle_rad);
        this->screenHeightScaled11 = -this->screenWidthScaled*sin(rotationAngle_rad)+this->screenHeightScaled*cos(rotationAngle_rad);
        this->screenWidthScaled11 = screenWidthScaled11*screenHeight/screenWidth;
        //vertex10 top right
        this->screenWidthScaled10 = this->screenWidthScaled*cos(rotationAngle_rad)-this->screenHeightScaled*sin(rotationAngle_rad);
        this->screenHeightScaled10 = this->screenWidthScaled*sin(rotationAngle_rad)+this->screenHeightScaled*cos(rotationAngle_rad);
        this->screenWidthScaled10 = screenWidthScaled10*screenHeight/screenWidth;
        //vertex00 botton right
        this->screenWidthScaled00 = this->screenWidthScaled*cos(rotationAngle_rad)+this->screenHeightScaled*sin(rotationAngle_rad);
        this->screenHeightScaled00 = this->screenWidthScaled*sin(rotationAngle_rad)-this->screenHeightScaled*cos(rotationAngle_rad);
        this->screenWidthScaled00 = screenWidthScaled00*screenHeight/screenWidth;
    }



    //recalculate marker coordinates
    this->slot_setMarkerPosition(this->markerPosition);
}

void GLWindow2D::mousePressEvent(QMouseEvent* event){
	this->mousePos = event->pos();
}

void GLWindow2D::mouseMoveEvent(QMouseEvent* event){
	if(event->buttons() & Qt::LeftButton && !this->panel->underMouse()){
		QPoint delta = (event->pos() - this->mousePos);
		int windowWidth = this->size().width();
		int windowHeight = this->size().height();
        this->xTranslation += 2.0*(float)delta.x()/((float)windowWidth);
        this->yTranslation += -2.0*(float)delta.y()/(float)windowHeight;
	}
	this->mousePos = event->pos();
}

void GLWindow2D::wheelEvent(QWheelEvent *event){
	if(!this->panel->underMouse()){
		QPoint numPixels = event->pixelDelta();
		QPoint numDegrees = event->angleDelta()/8;

		if (!numPixels.isNull()) {
		   this->scaleFactor += numPixels.y()/30.0;

		} else if (!numDegrees.isNull()) {
			QPoint numSteps = numDegrees/15;
			this->scaleFactor += (float)numSteps.y()/30.0;
			if(this->scaleFactor<0.05){
				this->scaleFactor = 0.05;
			}
		}
		event->accept();
	}
}

void GLWindow2D::mouseDoubleClickEvent(QMouseEvent *event){
	if(!this->panel->underMouse()){
		this->scaleFactor = 1.0;
		this->xTranslation = 0.0;
		this->yTranslation = 0.0;
	}
}

void GLWindow2D::enterEvent(QEvent *event){
	this->panel->setVisible(true);
}

void GLWindow2D::leaveEvent(QEvent *event){
	this->panel->setVisible(false);
}
void GLWindow2D::contextMenuEvent(QContextMenuEvent *event) {
	this->contextMenu->exec(event->globalPos());
}

ControlPanel2D::ControlPanel2D(QWidget *parent) : QWidget(parent){
	this->panel = new QWidget(parent);
	QPalette pal;
	pal.setColor(QPalette::Background, QColor(32,32,32,128));
	this->panel->setAutoFillBackground(true);
	this->panel->setPalette(pal);
	this->widgetLayout = new QHBoxLayout(this);
	this->widgetLayout->addWidget(this->panel);
	this->spinBoxAverage = new QSpinBox(this->panel);
	this->spinBoxFrame = new QSpinBox(this->panel);
	this->spinBoxFrame->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	this->doubleSpinBoxRotationAngle = new QDoubleSpinBox(this->panel);
	this->doubleSpinBoxRotationAngle->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	this->labelFrame = new QLabel(tr("Display frame:"), this->panel);
	this->labelRotationAngle = new QLabel(tr("Rotation:"), this->panel);
	this->slider = new QSlider(Qt::Horizontal, this->panel);
	this->slider->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);
	this->stringBoxFunctions = new StringSpinBox(this->panel);
	this->labelDisplayFunction = new QLabel(tr("Display technique:"), this->panel);
	this->labelDisplayFunctionFrames = new QLabel(tr("Frames:"), this->panel);
	this->layout = new QGridLayout(this->panel);
	this->layout->setContentsMargins(0,0,0,0);
	this->layout->setVerticalSpacing(1);
	this->layout->addWidget(this->labelFrame, 0, 0, 1, 1, Qt::AlignRight);
	this->layout->addWidget(this->slider, 0, 1, 1, 5);
	this->layout->setColumnStretch(4, 100); //set high stretch factor such that slider gets max available space
	this->layout->addWidget(this->spinBoxFrame, 0, 6, 1, 1, Qt::AlignLeft);
	this->layout->addWidget(this->labelDisplayFunction, 1, 0, 1, 1, Qt::AlignRight);
	this->layout->addWidget(this->stringBoxFunctions, 1, 1, 1, 1, Qt::AlignLeft);
	this->layout->addWidget(this->labelDisplayFunctionFrames, 1, 2, 1, 1, Qt::AlignRight);
	this->layout->addWidget(this->spinBoxAverage, 1, 3, 1, 1, Qt::AlignLeft);
	this->layout->addWidget(this->labelRotationAngle, 1, 5, 1, 1, Qt::AlignRight);
	this->layout->addWidget(this->doubleSpinBoxRotationAngle, 1, 6, 1, 1, Qt::AlignLeft);

	this->doubleSpinBoxRotationAngle->setMinimum(-360.0);
	this->doubleSpinBoxRotationAngle->setMaximum(360.0);
	this->doubleSpinBoxRotationAngle->setDecimals(1);
	this->doubleSpinBoxRotationAngle->setSingleStep(1.0);
	this->spinBoxAverage->setMinimum(1);
	this->spinBoxAverage->setSingleStep(1);
	this->spinBoxFrame->setMinimum(0);
	this->spinBoxFrame->setSingleStep(1);
	this->slider->setMinimum(0);
	QStringList displayFunctionOptions = { "Averaging", "MIP"}; //todo: think of better way to add available display techniques
	this->stringBoxFunctions->setStrings(displayFunctionOptions);

	connect(this->slider, &QSlider::valueChanged, this->spinBoxFrame, &QSpinBox::setValue);
	connect(this->spinBoxFrame, QOverload<int>::of(&QSpinBox::valueChanged), this->slider, &QSlider::setValue);
	connect(this->slider, &QSlider::valueChanged, this, &ControlPanel2D::updateDisplayFrameSettings);
	connect(this->spinBoxAverage, QOverload<int>::of(&QSpinBox::valueChanged), this, &ControlPanel2D::updateDisplayFrameSettings);
	connect(this->stringBoxFunctions, &StringSpinBox::indexChanged, this, &ControlPanel2D::updateDisplayFrameSettings);
}

ControlPanel2D::~ControlPanel2D()
{

}

void ControlPanel2D::setMaxFrame(unsigned int maxFrame){
	this->spinBoxFrame->setMaximum(maxFrame);
	this->slider->setMaximum(maxFrame);
}

void ControlPanel2D::setMaxAverage(unsigned int maxAverage){
	this->spinBoxAverage->setMaximum(maxAverage);
}

void ControlPanel2D::updateDisplayFrameSettings(){
	emit displayFrameSettingsChanged(this->spinBoxFrame->value(), this->spinBoxAverage->value(), this->stringBoxFunctions->getIndex());
}
