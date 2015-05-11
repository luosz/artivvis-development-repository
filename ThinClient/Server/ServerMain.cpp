#include"ServerRenderer.h"
#include <stdlib.h>
#include "ServerNetworkManager.h"

#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720

//#define SCREEN_WIDTH 640
//#define SCREEN_HEIGHT 360

VolumeRenderer volumeRenderer;
NetworkManager networkManager;
VolumeDataset volume;
unsigned char *pixelBuffer;

clock_t oldTime;

// For mouse control
int xclickedAt = -1;
int yclickedAt = -1;


void Init()
{
	pixelBuffer = new unsigned char[3 * SCREEN_WIDTH * SCREEN_HEIGHT];

	volume.Init();
	volumeRenderer.Init(SCREEN_WIDTH, SCREEN_HEIGHT, volume);
	networkManager.Init();

	oldTime = clock();
}

// Update renderer and check for Qt events
void MainLoop()
{
	volume.Update();
	volumeRenderer.Update();

	clock_t currentTime = clock();
	float time = (currentTime - oldTime) / (float) CLOCKS_PER_SEC;

	if (time > 0.1f)
	{
		glReadPixels(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, pixelBuffer);

		networkManager.Update(SCREEN_WIDTH, SCREEN_HEIGHT, pixelBuffer);

		oldTime = currentTime;
	}

	
}


// General Keyboard Input
void KeyboardFunc (unsigned char key, int xmouse, int ymouse)
{
	switch(key)
	{

		case 27:
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

		
	Init();

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
