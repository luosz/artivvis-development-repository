#version 330

in vec2 vPosition;
in vec2 vTexture;

out vec2 texCoords;

void main()
{
	texCoords = vTexture;

	gl_Position = vec4 (vPosition,0.0, 1.0);
}