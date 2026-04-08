# version 330

in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;;
uniform mat4 p3d_ModelViewProjectionMatrix;
out vec3 v_position;
out vec2 texcoord;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_position = p3d_Vertex.xyz;
    texcoord = p3d_MultiTexCoord0;

}