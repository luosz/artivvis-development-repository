#include "Shader.h"

using namespace std;

//Reads in shader text files
void Shader::Load(string vName, string fName)
{
	ifstream readFileV;
	readFileV.open(vName);
	int i = 0;

	if (readFileV.is_open())
	{
		while(readFileV.good())
		{
			pVS[i] = readFileV.get();
			if (!readFileV.eof())
			{
				i++;
			}
			pVS[i] = 0;
		}
		readFileV.close();
	} else 
	{
		cout << "Unable to open shader file" << endl;
	}
	ifstream readFileF;
	readFileF.open(fName);
	i = 0;

	if (readFileF.is_open())
	{
		while(readFileF.good())
		{
			pFS[i] = readFileF.get();
			if (!readFileF.eof())
			{
				i++;
			}
			pFS[i] = 0;
		}
		readFileF.close();
	} else 
	{
		cout << "Unable to open shader file" << endl;
	}

	CompileShaders();
}

// Shader Functions- click on + to expand
#pragma region SHADER_FUNCTIONS
void Shader::AddShader(GLuint ShaderProgram, const char* pShaderText, GLenum ShaderType)
{
	// create a shader object
    GLuint ShaderObj = glCreateShader(ShaderType);

    if (ShaderObj == 0) 
	{
        fprintf(stderr, "Error creating shader type %d\n", ShaderType);
        exit(0);
    }
	// Bind the source code to the shader, this happens before compilation
	glShaderSource(ShaderObj, 1, (const GLchar**)&pShaderText, NULL);
	// compile the shader and check for errors
    glCompileShader(ShaderObj);
    GLint success;
	// check for shader related errors using glGetShaderiv
    glGetShaderiv(ShaderObj, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar InfoLog[1024];
        glGetShaderInfoLog(ShaderObj, 1024, NULL, InfoLog);
        fprintf(stderr, "Error compiling shader type %d: '%s'\n", ShaderType, InfoLog);
        exit(1);
    }
	// Attach the compiled shader object to the program object
    glAttachShader(ShaderProgram, ShaderObj);
}

GLuint Shader::CompileShaders()
{
	//Start the process of setting up our shaders by creating a program ID
	//Note: we will link all the shaders together into this ID
    ID = glCreateProgram();
    if (ID == 0) 
	{
        fprintf(stderr, "Error creating shader program\n");
        exit(1);
    }

	// Create two shader objects, one for the vertex, and one for the fragment shader
    AddShader(ID, pVS, GL_VERTEX_SHADER);
    AddShader(ID, pFS, GL_FRAGMENT_SHADER);

    GLint Success = 0;
    GLchar ErrorLog[1024] = { 0 };


	// After compiling all shader objects and attaching them to the program, we can finally link it
    glLinkProgram(ID);
	// check for program related errors using glGetProgramiv
    glGetProgramiv(ID, GL_LINK_STATUS, &Success);
	if (Success == 0) 
	{
		glGetProgramInfoLog(ID, sizeof(ErrorLog), NULL, ErrorLog);
		fprintf(stderr, "Error linking shader program: '%s'\n", ErrorLog);
        exit(1);
	}

	// program has been successfully linked but needs to be validated to check whether the program can execute given the current pipeline state
    glValidateProgram(ID);

	// check for program related errors using glGetProgramiv
    glGetProgramiv(ID, GL_VALIDATE_STATUS, &Success);
    if (!Success) 
	{
        glGetProgramInfoLog(ID, sizeof(ErrorLog), NULL, ErrorLog);
        fprintf(stderr, "Invalid shader program: '%s'\n", ErrorLog);
        exit(1);
    }

	// Finally, use the linked shader program
	// Note: this program will stay in effect for all draw calls until you replace it with another or explicitly disable its use
	return ID;
}
#pragma endregion SHADER_FUNCTIONS



void Shader::LoadCompute(string vName)
{
	ifstream readFileV;
	readFileV.open(vName);
	int i = 0;

	if (readFileV.is_open())
	{
		while(readFileV.good())
		{
			pVS[i] = readFileV.get();
			if (!readFileV.eof())
			{
				i++;
			}
			pVS[i] = 0;
		}
		readFileV.close();
	} else 
	{
		cout << "Unable to open shader file" << endl;
	}

	CompileCompute();
}




GLuint Shader::CompileCompute()
{
	//Start the process of setting up our shaders by creating a program ID
	//Note: we will link all the shaders together into this ID
    ID = glCreateProgram();
    if (ID == 0) 
	{
        fprintf(stderr, "Error creating shader program\n");
        exit(1);
    }

	// Create two shader objects, one for the vertex, and one for the fragment shader
    AddShader(ID, pVS, GL_VERTEX_SHADER);

    GLint Success = 0;
    GLchar ErrorLog[1024] = { 0 };


	// After compiling all shader objects and attaching them to the program, we can finally link it
    glLinkProgram(ID);
	// check for program related errors using glGetProgramiv
    glGetProgramiv(ID, GL_LINK_STATUS, &Success);
	if (Success == 0) 
	{
		glGetProgramInfoLog(ID, sizeof(ErrorLog), NULL, ErrorLog);
		fprintf(stderr, "Error linking shader program: '%s'\n", ErrorLog);
        exit(1);
	}

	// program has been successfully linked but needs to be validated to check whether the program can execute given the current pipeline state
    glValidateProgram(ID);

	// check for program related errors using glGetProgramiv
    glGetProgramiv(ID, GL_VALIDATE_STATUS, &Success);
    if (!Success) 
	{
        glGetProgramInfoLog(ID, sizeof(ErrorLog), NULL, ErrorLog);
        fprintf(stderr, "Invalid shader program: '%s'\n", ErrorLog);
        exit(1);
    }

	// Finally, use the linked shader program
	// Note: this program will stay in effect for all draw calls until you replace it with another or explicitly disable its use
	return ID;
}