#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>
#include "Renderer.h"

#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720


// For mouse control
struct MousePos
{
	int xclickedAt;
	int yclickedAt;

	MousePos()
	{
		xclickedAt = -1;
		yclickedAt = -1;
	}
};

MousePos leftMouse;
MousePos rightMouse;

Renderer renderer;

void Init()
{
	
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glEnable(GL_DEPTH_TEST);
	renderer.Init(SCREEN_WIDTH, SCREEN_HEIGHT);
}



// Update renderer and check for Qt events
void Update()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	renderer.Draw();

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
	glutPostRedisplay();
 }


// Mouse drags to control camera
void MouseMove(int x, int y) 
{ 	
	if (rightMouse.xclickedAt >= 0)
	{
		renderer.camera.OrbitSideways(rightMouse.xclickedAt - x);
		rightMouse.xclickedAt = x;
	}	
	if (rightMouse.yclickedAt >= 0)
	{
		renderer.camera.OrbitUp((y - rightMouse.yclickedAt) * 0.1f);
		rightMouse.yclickedAt = y;
	}
}


// Mouse clicks to initialize drags
void MouseButton(int button, int state, int x, int y) 
{
	if (button == GLUT_RIGHT_BUTTON) 
	{
		if (state == GLUT_UP)
		{
			rightMouse.xclickedAt = -1;
			rightMouse.yclickedAt = -1;
		}
		else
		{
			rightMouse.xclickedAt = x;
			rightMouse.yclickedAt = y;
		}
	}
}


// Mouse wheel to zoom camera
void MouseWheel(int wheel, int direction, int x, int y) 
{
	renderer.camera.OrbitZoom(-direction * 0.2f);
}




int main(int argc, char *argv[])
{
	// Set up the window
	glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB);
	glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);
    glutCreateWindow("Volume GI");

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
		
	// Initialize Renderer and Qt
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