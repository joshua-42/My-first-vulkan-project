#version 450

layout(location = 2) flat in vec3 fragNormal;
layout(location = 0) out vec4 outColor;

void main() {
    // Normalise la normale
    vec3 n = normalize(fragNormal);

    // Calcule une intensité à partir de la normale (par exemple : axe Z)
    float intensity = n.z * 0.5 + 0.5; // transforme [-1,1] → [0,1]

    // Utilise cette intensité pour faire un gris uniforme
    outColor = vec4(vec3(intensity), 1.0);
}
