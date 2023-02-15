#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 setColor;
out vec3 Normal;
out vec3 FragPos;
out vec3 lightPos;
void main()
{
	gl_Position = projection * view * model * vec4(aPos, 1.0f);
	setColor = aColor;
	Normal = mat3(transpose(inverse(view))) * aNormal;
	FragPos = vec3(view*vec4(aPos,1.0));
	lightPos = vec3(-1000,1000,1000);
}