#include"VolumeRenderer.h"
#include "QtWidget.h"
#include <QtWidgets/QApplication>
#include "../TransferFunctionEditor/import_transfer_function_editor.h"

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 800

VolumeRenderer volumeRenderer;

// For mouse control
int xclickedAt = -1;
int yclickedAt = -1;


// Update renderer and check for Qt events
void MainLoop()
{
	volumeRenderer.Update();

	QCoreApplication::processEvents();
}


// General Keyboard Input
void KeyboardFunc (unsigned char key, int xmouse, int ymouse)
{
	switch(key)
	{

		case 27:
			cudaDeviceReset();
			exit(0);
			break;
	}
}


// Keyboad Special Keys Input
void SpecialFunc(int key, int x, int y)
 {
	switch(key)
	{
	case GLUT_KEY_UP:
		volumeRenderer.camera.position.z -= 0.5f;
		break;
	case GLUT_KEY_LEFT:
		volumeRenderer.camera.position.x -= 0.5f;
		break;
	case GLUT_KEY_DOWN:
		volumeRenderer.camera.position.z += 0.5f;
		break;
	case GLUT_KEY_RIGHT:
		volumeRenderer.camera.position.x += 0.5f;
		break;
	case GLUT_KEY_PAGE_UP:
		volumeRenderer.camera.position.y += 0.5f;
		break;
	case GLUT_KEY_PAGE_DOWN:
		volumeRenderer.camera.position.y -= 0.5f;
		break;
	}
	glutPostRedisplay();
 }


// Mouse drags to control camera
void MouseMove(int x, int y) 
{ 	
	if (xclickedAt >= 0)
	{
		volumeRenderer.camera.Rotate((float)(xclickedAt - x));
		xclickedAt = x;
	}
		

	if (yclickedAt >= 0)
	{
		volumeRenderer.camera.Translate(glm::vec3(0.0f, (y - yclickedAt) * 0.1f, 0.0f));
		yclickedAt = y;
	}
}


// Mouse clicks to initialize drags
void MouseButton(int button, int state, int x, int y) 
{
	if (button == GLUT_RIGHT_BUTTON) 
	{
		if (state == GLUT_UP)
		{
			xclickedAt = -1;
			yclickedAt = -1;
		}
		else
		{
			xclickedAt = x;
			yclickedAt = y;
		}
	}

	if (volumeRenderer.grabRegion)
	{
		if (button == GLUT_LEFT_BUTTON)
		{
			if (state == GLUT_DOWN)
				volumeRenderer.OptimizeForSelectedRegion(x, y, SCREEN_WIDTH, SCREEN_HEIGHT);
		}
	}
}


// Mouse wheel to zoom camera
void MouseWheel(int wheel, int direction, int x, int y) 
{
	if (volumeRenderer.grabRegion)
		volumeRenderer.renderer->raycaster->clipPlaneDistance += (direction * 0.02f);
	else
		volumeRenderer.camera.Zoom(-direction * 0.2f);	
}



int main(int argc, char *argv[])
{
	
	// Set up the window
	glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB);
	glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);
    glutCreateWindow("Volume Renderer");

	// Tell glut where the display function is
	glutDisplayFunc(MainLoop);
	glutIdleFunc(MainLoop);

	// A call to glewInit() must be done after glut is initialized!
    GLenum res = glewInit();
	// Check for any errors
    if (res != GLEW_OK) 
	{
		fprintf(stderr, "Error: '%s'\n", glewGetErrorString(res));
		return 1;
    }

	// Initialize Qt Widget, QApplication must be called before any Qt operations are performed
	QApplication a(argc, argv);
	QtWidget w;
	w.show();
		
	// Initialize Renderer and Qt
	volumeRenderer.Init(SCREEN_WIDTH, SCREEN_HEIGHT);	
	w.Init(volumeRenderer);

	// Initialize Transfer Function Editor
	TransferFunctionEditor tf;
	tf.init(volumeRenderer);
	tf.show();
	volumeRenderer.renderer->transferFunction.tfView = &tf.tf;

	// Specify glut input functions
	glutKeyboardFunc(KeyboardFunc);
	glutSpecialFunc(SpecialFunc);
	glutMouseFunc(MouseButton);
	glutMotionFunc(MouseMove);
	glutMouseWheelFunc(MouseWheel);

	// Begin infinite event loop
	glutMainLoop();
	
    return 0;

}
