#include "Camera.h"
#include "StaticVolume.h"
#include "ShaderManager.h"

#define SCREEN_WIDTH 400
#define SCREEN_HEIGHT 400

Camera camera;
GLuint shaderProgramID;

ShaderType currentShader;
ShaderManager shaderManager;

StaticVolume staticVolume;

// For mouse control
int xclickedAt = -1;
int yclickedAt = -1;

GLuint texId;


void Init()
{	
	glEnable(GL_DEPTH_TEST);

	//Set up camera
	camera.Init(SCREEN_WIDTH, SCREEN_HEIGHT);

	// Read in volume from File and 
	staticVolume.ReadFiles();	

	// Initializing shaders for OpenGL stuff
	shaderManager.Init();
	currentShader = RaycastShader;

	// Initialize 2D texture to draw to
	glGenTextures(1, &texId);
    glBindTexture(GL_TEXTURE_2D, texId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SCREEN_WIDTH, SCREEN_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL); 
	glBindTexture(GL_TEXTURE_2D, 0);
}


// Calculate normal using gradient at voxel
glm::vec3 CalculateNormal(int x, int y, int z)
{
	// Samples position diagonal and negative diagonal
	glm::vec3 sample1, sample2;

	if (x == 0 || y == 0 || z == 0 || x == staticVolume.xRes-1 || y == staticVolume.yRes-1 || z == staticVolume.zRes-1)
		return glm::vec3(0.0f);

	int index = (x-1) + (y * staticVolume.xRes) + (z * staticVolume.xRes * staticVolume.yRes);
	sample1.x = staticVolume.shorts[index];
	index = (x+1) + (y * staticVolume.xRes) + (z * staticVolume.xRes * staticVolume.yRes);
	sample2.x = staticVolume.shorts[index];
	index = x + ((y-1) * staticVolume.xRes) + (z * staticVolume.xRes * staticVolume.yRes);
	sample1.y = staticVolume.shorts[index];
	index = x + ((y+1) * staticVolume.xRes) + (z * staticVolume.xRes * staticVolume.yRes);
	sample2.y = staticVolume.shorts[index];
	index = x + (y * staticVolume.xRes) + ((z-1) * staticVolume.xRes * staticVolume.yRes);
	sample1.z = staticVolume.shorts[index];
	index = x + (y * staticVolume.xRes) + ((z+1) * staticVolume.xRes * staticVolume.yRes);
	sample2.z = staticVolume.shorts[index];

	return glm::normalize(sample2 - sample1);
}

// Lambertian lighting calculation using normal information
glm::vec4 CalculateLighting(glm::vec4 color, glm::vec3 N, glm::vec3 rayPosition)
{
	glm::vec3 lightDirection = glm::vec3(1.0f, 1.0f, 1.0f);
	glm::vec4 diffuseLight = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);
	glm::vec4 specularLight = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);

	glm::vec3 L, H;

	L = glm::normalize(lightDirection);
	H = glm::normalize(L + glm::normalize(-rayPosition));
	
	float diff = glm::clamp(glm::dot(N,L), 0.0f, 1.0f);
	glm::vec4 spec = specularLight * pow (glm::max(0.0f, glm::dot(H, N)), 50.0f); 

	return ((color * diff) + spec);
}

// CPU based raycast
void CPUraycast()
{
	// Step length of ray
	float stepsize = 0.01f;

	// Max steps a ray can take
	int maxRaySteps = 800;

	// Division 1 is any voxel between this value and div2
	short div1 = 500;
	// Division 2 is any voxel over this value
	short div2 = 1500;

	// Limits of the box surrounding the volume
	float xMin, yMin, zMin, xMax, yMax, zMax;
	xMin = yMin = zMin = -1.0f;
	xMax = yMax = zMax = 1.0f;

	// Used for termination later
	bool entered = false;

	// Calculations to find image plane that is perpendicular to camera view
	glm::vec3 camDirection = glm::normalize(camera.focus - camera.position);
	glm::vec3 rightVec = glm::normalize(glm::cross(camDirection, glm::vec3(0.0f, 1.0f, 0.0f)));
	glm::vec3 upVec = glm::normalize(glm::cross(camDirection, -rightVec));

	// Use camera field of view to find extents of plane
	float extent = glm::tan((camera.FoV / 2.0f) * (glm::pi<float>()/180.0f));

	// Use these previously calculated results to find upper left corner of image plane which corresponds to upper left pixel
	glm::vec3 topLeft;

	glm::vec3 temp = camera.position + camDirection;
	temp = temp + (extent * upVec);
	topLeft = temp - (extent * rightVec);
	
	float delta = (extent * 2.0f) / (float)SCREEN_WIDTH;
	
	// This will be the vector which will hold the calculated colour information
	std::vector<float> scalars;
	scalars.resize(SCREEN_HEIGHT*SCREEN_WIDTH*4);

	// Set opecities for each division
	float opacity1 = 0.4f;
	float opacity2 = 0.6f;
	
	// Move an amount of pixel widths/heights away from the corner to raycast at another pixel
	for (int y=0; y<SCREEN_HEIGHT; y++)
	{
		for (int x=0; x<SCREEN_WIDTH; x++)
		{
			// Calculate new pixel position on image plane
			glm::vec3 pixelPos = glm::vec3(topLeft + ((delta * x) * rightVec) - ((delta * y) * upVec));

			// Cast ray in this direction
			glm::vec3 rayDir = glm::normalize(pixelPos - camera.position);

			// Initialize ray

			// Hasn't enetered the box yet
			entered = false;

			// Full transmittance so far
			float absorption = 0.0f;

			// Hasn't hit either isosurface we are looking for
			bool hitDiv1 = false;
			bool hitDiv2 = false;

			// Colour set to black
			glm::vec4 finalColor = glm::vec4(0.0f);
			glm::vec4 color = glm::vec4(0.0f);
			glm::vec3 normal = glm::vec3(0.0f);

			// Step along the ray from camera
			for (int i=0; i<maxRaySteps; i++)
			{
				// New sampling position
				glm::vec3 rayPosition = camera.position + (rayDir*(float)i*stepsize);
			
				// If its inside the bounds of the box
				if (glm::abs(rayPosition.x) < 1.0f && glm::abs(rayPosition.y) < 1.0f && glm::abs(rayPosition.z) < 1.0f && absorption < 1.0f)
				{
					entered = true;

					// Find discrete voxel index
					int discreteX = (rayPosition.x - xMin) / (2.0f / (float)staticVolume.xRes);
					int discreteY = (rayPosition.y - yMin) / (2.0f / (float)staticVolume.yRes);
					int discreteZ = (rayPosition.z - zMin) / (2.0f / (float)staticVolume.zRes);

					if (discreteX < staticVolume.xRes && discreteY < staticVolume.yRes && discreteZ < staticVolume.zRes)
					{
						int index = discreteX + (discreteY * staticVolume.xRes) + (discreteZ * staticVolume.xRes * staticVolume.yRes);
						short value = staticVolume.shorts[index];

						// If the sampled value falls within the range we are looking for and hasn't been here before
						if (value > div1 && value < div2 && !hitDiv1)	
						{			
							// Add a green colour affected by lighting calculations
							color = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
							normal = CalculateNormal(discreteX, discreteY, discreteZ);

							color = CalculateLighting(color, normal, rayPosition);

							// Check current absorption of the ray to see how much this surface contributes depending on its opacity
							if ((absorption + opacity1) > 1.0f)
								color = (1.0f - absorption) * color;
							else
								color = opacity1 * color;

							// Update colour
							finalColor += color;
							absorption += opacity1;
							
							hitDiv1 = true;
						}	
						// Same again for second range of value
						else if (value > div2 && !hitDiv2)
						{
							// Red affected by light
							color = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
							normal = CalculateNormal(discreteX, discreteY, discreteZ);

							color = CalculateLighting(color, normal, rayPosition);

							if ((absorption + opacity1) > 1.0f)
								color = (1.0f - absorption) * color;
							else
								color = opacity1 * color;

							finalColor += color;
							absorption += opacity1;

							hitDiv2 = true;
						}
					}
				}
				else
				{
					// If it was in the box and no longer is we know we have reached far side and can terminate
					if (entered)
						break;
				}
			}

			// Add final colour accumulation to the final vector
			scalars[((x + (y *SCREEN_WIDTH))*4)] = finalColor.x;
			scalars[((x + (y *SCREEN_WIDTH))*4)+1] = finalColor.y;
			scalars[((x + (y *SCREEN_WIDTH))*4)+2] = finalColor.z;
			scalars[((x + (y *SCREEN_WIDTH))*4)+3] = 1.0f;
		}
	}
	

	shaderProgramID = shaderManager.UseShader(TextureShader);

	// Copy the colour vector to a 2D texture for rendering
	glActiveTexture (GL_TEXTURE0);
	int texLoc = glGetUniformLocation(shaderProgramID,"texColor");
	glUniform1i(texLoc,0);
	glBindTexture (GL_TEXTURE_2D, texId);
	glTexSubImage2D(GL_TEXTURE_2D, 0,0,0,SCREEN_WIDTH, SCREEN_HEIGHT, GL_RGBA, GL_FLOAT, &scalars[0]);
	
	int texcoords_location = glGetAttribLocation (shaderProgramID, "vTexture");

	// Render 2D texture as a screen aligned quad using simple shader
	glBegin(GL_QUADS);
	glVertexAttrib2f(texcoords_location, 1.0f, 1.0f);
	glVertex2f(1.0f, 1.0f);

	glVertexAttrib2f(texcoords_location, 0.0f, 1.0f);
	glVertex2f(-1.0f, 1.0f);

	glVertexAttrib2f(texcoords_location, 0.0f, 0.0f);
	glVertex2f(-1.0f, -1.0f);

	glVertexAttrib2f(texcoords_location, 1.0f, 0.0f);
	glVertex2f(1.0f, -1.0f);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, 0);
}




void Update()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(0.0, 0.0, 0.0, 0.0);

	camera.Update();

	shaderProgramID = shaderManager.UseShader(currentShader);

	CPUraycast();

	glutSwapBuffers();
}





void KeyboardFunc (unsigned char key, int xmouse, int ymouse)
{
	
	switch(key)
	{

	}
}

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


void MouseMove(int x, int y) 
{ 	
	if (xclickedAt >= 0)
	{
		camera.position = glm::rotateY(camera.position, (float)(xclickedAt - x));
		xclickedAt = x;
	}
		

	if (yclickedAt >= 0)
	{
		camera.position.y += ((y - yclickedAt) * 0.1f);
		camera.position = camera.focus + (glm::normalize(camera.position - camera.focus) * camera.distFromFocus);
		yclickedAt = y;
	}
}

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


void MouseWheel(int wheel, int direction, int x, int y) 
{
	if (direction == 1)
		camera.distFromFocus -= 0.2f;
	else if (direction == -1)
		camera.distFromFocus += 0.2f;

	camera.position = camera.focus + (glm::normalize(camera.position - camera.focus) * camera.distFromFocus);
}


int main(int argc, char** argv)
{
	// Set up the window
	glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB);
	glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);
    glutCreateWindow("CPU Based Raycast");
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

	// Set up your objects and shaders
	Init();
	// Begin infinite event loop
	
	glutKeyboardFunc(KeyboardFunc);
	glutSpecialFunc(SpecialFunc);
	glutMouseFunc(MouseButton);
	glutMotionFunc(MouseMove);
	glutMouseWheelFunc(MouseWheel);
	glutMainLoop();
    return 0;
}