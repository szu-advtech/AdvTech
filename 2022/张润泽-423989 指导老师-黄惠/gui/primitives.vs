#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec4 Color;

void main()
{
	gl_Position = projection * view * model * vec4(aPos, 1.0f);
	Color = Color+vec4(1.0,1.0,1.0,0.0);
	Color = Color/2;
}