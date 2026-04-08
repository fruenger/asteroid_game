#version 430
in vec4 p3d_Vertex;                         // Panda3D vertex attribute
uniform mat4 p3d_ModelViewProjectionMatrix; // Panda3D MVP
out vec3 v_model_pos;                       // forwarded to fragment shader

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_model_pos = p3d_Vertex.xyz;
}