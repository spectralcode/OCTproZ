#version 430

in vec4 a_position;

uniform mat4 ModelViewProjectionMatrix;

void main() {
	gl_Position = ModelViewProjectionMatrix * a_position;
}
