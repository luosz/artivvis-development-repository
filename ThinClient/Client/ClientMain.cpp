#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>
#include "ClientNetworkManager.h"
#include "Texture2D.h"
#include "ShaderManager.h"

#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720

//#define SCREEN_WIDTH 640
//#define SCREEN_HEIGHT 360

int xclickedAt = -1;
int yclickedAt = -1;

NetworkManager networkManager;
unsigned char *pixelBuffer;
Texture2D *renderTex;



void Init()
{
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glEnable(GL_DEPTH_TEST);

	pixelBuffer = new unsigned char[4 * SCREEN_WIDTH * SCREEN_HEIGHT];
	renderTex = new Texture2D(GL_RGB, SCREEN_WIDTH, SCREEN_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, NULL);

	networkManager.Init(SCREEN_WIDTH, SCREEN_HEIGHT, pixelBuffer);

	ShaderManager::Init();
}

void Update()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBindTexture(GL_TEXTURE_2D, renderTex->ID);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, renderTex->xPixels, renderTex->yPixels, GL_RGB, GL_UNSIGNED_BYTE, pixelBuffer);
	glBindTexture(GL_TEXTURE_2D, 0);

	renderTex->Render();

	glutSwapBuffers();
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
 }


// Mouse drags to control camera
void MouseMove(int x, int y) 
{ 	
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
	
}

void Idle()
{
	glutPostRedisplay();
}

int main(int argc, char *argv[])
{
	// Set up the window
	glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB);
	glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);
    glutCreateWindow("Volume Renderer Client");


	// Tell glut where the display function is
	glutDisplayFunc(Update);
	glutIdleFunc(Update);

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
	glutMotionFunc(NULL);
	glutMouseWheelFunc(MouseWheel);
	glutPassiveMotionFunc(NULL);

	// Begin infinite event loop
	glutMainLoop();
	
    return 0;

}