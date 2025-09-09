#version 450

// UBO contenant les matrices
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

// Entrées vertex (positions, normales, etc.)
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;      // pas utilisé ici
layout(location = 2) in vec2 inTexCoord;   // pas utilisé ici
layout(location = 3) in vec3 inNormal;

// Sorties vers le fragment shader
layout(location = 2) flat out vec3 fragNormal; // flat shading → pas d'interpolation

void main() {
    // Position finale du sommet
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);

    // Transforme la normale dans l’espace monde (évite les bugs si model ≠ identité)
    mat3 normalMatrix = transpose(inverse(mat3(ubo.model)));
    fragNormal = normalize(normalMatrix * inNormal);
}
