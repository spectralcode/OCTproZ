#ifndef GLWINDOW_H
#define GLWINDOW_H


#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QMouseEvent>
#include <kernels.h>

// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_profiler_api.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// CUDA utilities
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

// Helper functions
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>



#include <QTime>



class GLWindow : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    GLWindow(QWidget* parent);
    ~GLWindow();

private:
    void* cudaBufHandle;

    QSize imgSize;
    GLuint buf;
    GLuint texture;

    bool m_core;
    int m_xRot;
    int m_yRot;
    int m_zRot;
    QPoint m_lastPos;


    QTime time;
    size_t frameCnt;

    unsigned short* testbuffer;
    float* testFloatBuffer;

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int width, int height) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;


public slots:
    void setXRotation(int angle);
    void setYRotation(int angle);
    void setZRotation(int angle);
    void cleanup();


signals:
    void xRotationChanged(int angle);
    void yRotationChanged(int angle);
    void zRotationChanged(int angle);

    void fps(double fps);
    void error(QString);
    void info(QString);

};

#endif // GLWINDOW_H























