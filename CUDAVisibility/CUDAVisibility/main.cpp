#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>
#include "Camera.h"
#include "ShaderManager.h"
#include "GPURaycaster.h"
#include "TransferFunction.h"
#include "VolumeDataset.h"
#include "VisibilityHist.h"

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 800


Camera camera;
ShaderManager shaderManager;
GPURaycaster raycaster;
TransferFunction transferFunction;
VolumeDataset volume;
VisibilityHistogram visibilityHistogram;

// For mouse control
int xclickedAt = -1;
int yclickedAt = -1;

bool showGraph;

void Init()
{
	cudaGLSetGLDevice(0);

	volume.Init();
	volume.currTexture3D = volume.GenerateTexture();

	camera.Init(SCREEN_WIDTH, SCREEN_HEIGHT);
	shaderManager.Init();

	transferFunction.Init(" ");

	visibilityHistogram.Init(SCREEN_WIDTH, SCREEN_HEIGHT);
	raycaster.Init(SCREEN_WIDTH, SCREEN_HEIGHT);

	showGraph = true;
}





// Update renderer and check for Qt events
void Update()
{
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	camera.Update();

	visibilityHistogram.CalculateHistogram(volume, transferFunction, shaderManager, camera);

	glEnable(GL_DEPTH_TEST);

	if (showGraph)
		visibilityHistogram.DrawHistogram(shaderManager, camera);
	else
	{
		GLuint shaderProgramID = shaderManager.UseShader(TransFuncShader);
		raycaster.Raycast(volume, transferFunction, shaderProgramID, camera);
	}


	glutSwapBuffers();
}




// General Keyboard Input
void KeyboardFunc (unsigned char key, int xmouse, int ymouse)
{
	switch(key)
	{
		case 'm':
			visibilityHistogram.CalculateHistogram(volume, transferFunction, shaderManager, camera);
			break;
		case 'n':
			if (showGraph)
				showGraph = false;
			else
				showGraph = true;
			break;
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
		camera.position.z -= 0.5f;
		break;
	case GLUT_KEY_LEFT:
		camera.position.x -= 0.5f;
		break;
	case GLUT_KEY_DOWN:
		camera.position.z += 0.5f;
		break;
	case GLUT_KEY_RIGHT:
		camera.position.x += 0.5f;
		break;
	case GLUT_KEY_PAGE_UP:
		camera.position.y += 0.5f;
		break;
	case GLUT_KEY_PAGE_DOWN:
		camera.position.y -= 0.5f;
		break;
	}
	glutPostRedisplay();
 }


// Mouse drags to control camera
void MouseMove(int x, int y) 
{ 	
	if (xclickedAt >= 0)
	{
		camera.Rotate((float)(xclickedAt - x));
		xclickedAt = x;
	}
		

	if (yclickedAt >= 0)
	{
		camera.Translate(glm::vec3(0.0f, (y - yclickedAt) * 0.1f, 0.0f));
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
	camera.Zoom(-direction * 0.2f);	
}




int main(int argc, char *argv[])
{
	
	// Set up the window
	glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB);
	glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);
    glutCreateWindow("Visibility Histogram");
		
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