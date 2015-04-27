#include"VolumeRenderer.h"
#include "CudaHeaders.h"
#include <stdlib.h>
#include "ServerNetworkManager.h"

#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720

VolumeRenderer volumeRenderer;
NetworkManager networkManager;

// For mouse control
int xclickedAt = -1;
int yclickedAt = -1;


// Update renderer and check for Qt events
void MainLoop()
{
	volumeRenderer.Update();
	networkManager.Update();
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
}


// Mouse wheel to zoom camera
void MouseWheel(int wheel, int direction, int x, int y) 
{
	volumeRenderer.camera.Zoom(-direction * 0.2f);	
}

int main(int argc, char *argv[])
{
	// Set up the window
	glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA);
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

		
	volumeRenderer.Init(SCREEN_WIDTH, SCREEN_HEIGHT);
	networkManager.Init(&volumeRenderer);


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
