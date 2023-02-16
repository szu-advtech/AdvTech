#version 330 core
out vec4 FragColor;  
in vec3 setColor;
in vec3 Normal;
in vec3 FragPos;
in vec3 lightPos;

void main()
{
	vec3 lightColor = vec3(1,1,1);
    	float ambientStrength = 0.3;
    	vec3 ambient = ambientStrength*lightColor;
	vec3 lightDir = normalize(lightPos - FragPos);
	float diff = max(dot(normalize(Normal),lightDir),0.0);
	vec3 diffuse = diff*lightColor;
	vec3 result = (ambient+diffuse)*setColor;
	FragColor = vec4(result, 1.0);

}