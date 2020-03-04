#include "glwindow.h"
#include "kernels.h"






GLWindow::GLWindow(QWidget *parent) : QOpenGLWidget(parent)
{

    m_core = QSurfaceFormat::defaultFormat().profile() == QSurfaceFormat::CoreProfile;
    // --transparent causes the clear color to be transparent. Therefore, on systems that
    // support it, the widget will become transparent apart from the logo.
  /*  if (m_transparent) {
        QSurfaceFormat fmt = format();
        fmt.setAlphaBufferSize(8);
        setFormat(fmt);
    }*/
   // this->initializeGL();
}

GLWindow::~GLWindow()
{
    cleanup();
}





static void qNormalizeAngle(int &angle)
{
    while (angle < 0)
        angle += 360 * 16;
    while (angle > 360 * 16)
        angle -= 360 * 16;
}

void GLWindow::setXRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != m_xRot) {
        m_xRot = angle;
        emit xRotationChanged(angle);
        update();
    }
}

void GLWindow::setYRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != m_yRot) {
        m_yRot = angle;
        emit yRotationChanged(angle);
        update();
    }
}

void GLWindow::setZRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != m_zRot) {
        m_zRot = angle;
        emit zRotationChanged(angle);
        update();
    }
}

void GLWindow::cleanup()
{
  /*  if (m_program == nullptr)
        return;
    makeCurrent();
    m_logoVbo.destroy();
    delete m_program;
    m_program = 0;
    doneCurrent();*/
}







#include <QDir>
#include <QCoreApplication>









void GLWindow::initializeGL()
{
    initializeOpenGLFunctions();

    CUDA_init();


















    QDir volumeDir = QDir(qApp->applicationDirPath());
    volumeDir.cdUp();
    volumeDir.cd("testVolume");
    QString fileName = volumeDir.absolutePath() + "/all";

    auto file = fopen(fileName.toLatin1(), "r");

    if(file == NULL){
        emit error("Unable to open file");
    }

    //get file size in units of sizeof(unsigend short)
    fseek(file, 0, SEEK_END); //seek end of file
    int fileLength = ftell(file);///(int)sizeof(unsigned short);   //end position within file (= file size) divides by size of ushort
    rewind(file);   //go back to start position of file


    this->testbuffer = (unsigned short*)valloc(fileLength);
    this->testFloatBuffer = (float*)valloc((fileLength/(int)sizeof(unsigned short))*sizeof(float));


    fread(this->testbuffer, sizeof(unsigned short), 512*2048, file);

    float max = 0;
    float coeff = 1/(pow(2.0f, 12)-1);
    for(int i = 0; i < 512*2048; i++){
        this->testFloatBuffer[i] = (float)(this->testbuffer[i])*coeff;
        float tt = this->testFloatBuffer[i];
        max = tt > max ? tt : max;
    }
    emit info(QString::number(max));

    fclose(file);








    QImage img("/home/oct/Bilder/balboa.jpg");
    imgSize = img.size();
    img = img.convertToFormat(QImage::Format_RGB32); // BGRA on little endian

    int width = 2048;
    int height = 512;

    glGenBuffers(1, &buf);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, buf);
    //glBufferData(GL_PIXEL_UNPACK_BUFFER, imgSize.width() * imgSize.height() * 4, img.constBits(), GL_DYNAMIC_COPY);
    //glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,  width * height * sizeof(unsigned short), this->testbuffer, GL_DYNAMIC_COPY);//GL_STREAM_DRAW_ARB);//GL_DYNAMIC_COPY);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,  width * height * sizeof(float), this->testFloatBuffer, GL_DYNAMIC_COPY);
    //glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaBufHandle = CUDA_registerBuffer(buf);

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, imgSize.width(), imgSize.height(), 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_SHORT, 0);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, 0); //This should work with 12bit unsigned short array but it does not
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    //this->time.start();
    //this->frameCnt = 0;
}


void GLWindow::paintGL()
{
    //----test
    //this->frameCnt += 1;
    //if(this->time.elapsed() >= 1000){
     //   double fps = (this->frameCnt/((double)time.elapsed()/1000.0));
     //   emit info(QString::number(fps));
    //}

    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, 0.0f);
    glScalef(1.0, 1.0, 0.f);

    //------test ende



    //glClearColor(1,0,0,1);
    //glClear(GL_COLOR_BUFFER_BIT);

    //glDisable(GL_DEPTH_TEST);

    int width = imgSize.width();//512;
    int height = imgSize.height();//1024;

    width = 2048;
    height = 512;

    //void *devPtr = CUDA_map(cudaBufHandle);
    //CUDA_do_something(devPtr, width, height);
    //CUDA_unmap(cudaBufHandle);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, buf);
    glBindTexture(GL_TEXTURE_2D, texture);
    // Fast path due to BGRA
    //glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imgSize.width(), imgSize.height(), GL_BGRA, GL_UNSIGNED_BYTE, 0);
    //glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    //glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, 0); //This should work with 12bit unsigned short array but it does not
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
        glTexCoord2f(0, 1), glVertex2f(-1.0, -1.0);
        glTexCoord2f(1, 1), glVertex2f(-1.0, 1.0);
        glTexCoord2f(1, 0), glVertex2f(1.0, 1.0);
        glTexCoord2f(0, 0), glVertex2f(1.0, -1.0);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    update(); // request the next frame
}

void GLWindow::resizeGL(int w, int h)
{
    //m_proj.setToIdentity();
    //m_proj.perspective(45.0f, GLfloat(w) / h, 0.01f, 100.0f);
}

void GLWindow::mousePressEvent(QMouseEvent *event)
{
   m_lastPos = event->pos();
}

void GLWindow::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->x() - m_lastPos.x();
    int dy = event->y() - m_lastPos.y();

    if (event->buttons() & Qt::LeftButton) {
        setXRotation(m_xRot + 8 * dy);
        setYRotation(m_yRot + 8 * dx);
    } else if (event->buttons() & Qt::RightButton) {
        setXRotation(m_xRot + 8 * dy);
        setZRotation(m_zRot + 8 * dx);
    }
    m_lastPos = event->pos();
}
