#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>
#include "Fluid.h"
#include "Camera.h"
#include "ShaderManager.h"
#include "GPURaycaster.h"

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 800

Fluid fluid;
Camera camera;
ShaderManager shaderManager;
GPURaycaster raycaster;

// For mouse control
int xclickedAt = -1;
int yclickedAt = -1;

int numFiles = 1;
int rawDataSize;
char* buffer;
char *pixelBuffer;


void Init()
{
	camera.Init(SCREEN_WIDTH, SCREEN_HEIGHT);
	shaderManager.Init();
	raycaster.Init(SCREEN_WIDTH, SCREEN_HEIGHT);

	fluid.Init();

	rawDataSize = gridXRes * gridYRes * gridZRes * sizeof(float);
	buffer = new char[rawDataSize];
	pixelBuffer = new char[SCREEN_WIDTH * SCREEN_HEIGHT * 4];
}

void CopyToFile()
{
	std::string targetFileName = "../../Samples/SmokeSim/SmokeSimRaw/smokeSim.";

	memcpy(buffer, &fluid.hostDensities[0], rawDataSize);

	ofstream outStream(targetFileName + to_string(numFiles), std::ios::out|std::ios::binary);

	if (outStream.is_open())
	{
		outStream.write (buffer, rawDataSize);
		outStream.close();
	}

	if (numFiles == 100)
	{
		cudaDeviceReset();
		exit(0);
	}

	numFiles++;
}

void CopyToVideoFile()
{
	std::string targetFileName = "../../Samples/SmokeSim/SmokeVideo.raw";

	ofstream outStream(targetFileName, std::ios::out|std::ios::binary|std::ios::ate);

	if (outStream.is_open())
	{
		outStream.write (pixelBuffer, SCREEN_WIDTH * SCREEN_HEIGHT * 4);
		outStream.close();
	}
}


// Update renderer and check for Qt events
void Update()
{
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	camera.Update();

	fluid.Update();

//	CopyToFile();

	glEnable(GL_DEPTH_TEST);
	GLuint shaderProgramID = shaderManager.UseShader(SmokeShader);
	raycaster.Raycast(shaderProgramID, camera, fluid.hostDensities, fluid.hostTemperatures);

	glutSwapBuffers();

//	glReadBuffer(GL_FRONT);
//	glPixelStorei(GL_PACK_ALIGNMENT, 4);
//	glPixelStorei(GL_PACK_ROW_LENGTH, 0);
//	glPixelStorei(GL_PACK_SKIP_ROWS, 0);
//	glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
//	glReadPixels(0,0,SCREEN_WIDTH, SCREEN_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, pixelBuffer);
//
//	CopyToVideoFile();
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
    glutCreateWindow("Fluid Simulation");

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