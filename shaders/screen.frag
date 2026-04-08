# version 330

in vec3 v_position;
in vec2 texcoord;
uniform float f_time;
uniform float[2] asteroid_pos;
uniform float asteroid_brightness;
out vec4 fragColor;

// Hash helpers
float hash(float n) { return fract(sin(n) * 43758.5453123); }
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }
float hash(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453123); }
float hash(vec4 p) { return fract(sin(dot(p, vec4(127.1, 311.7, 74.7, 19.3))) * 43758.5453123); }

// Gaussian blur function (3x3 kernel)
vec4 gaussianBlur(sampler2D tex, vec2 uv, vec2 texel_size) {
    float kernel[9] = float[9](
        1.0/16.0, 2.0/16.0, 1.0/16.0,
        2.0/16.0, 4.0/16.0, 2.0/16.0,
        1.0/16.0, 2.0/16.0, 1.0/16.0
    );

    vec2 offset[9] = vec2[9](
        vec2(-1, -1), vec2(0, -1), vec2(1, -1),
        vec2(-1,  0), vec2(0,  0), vec2(1,  0),
        vec2(-1,  1), vec2(0,  1), vec2(1,  1)
    );

    vec4 sum = vec4(0.0);
    for (int i = 0; i < 9; ++i) {
        vec2 sample_uv = uv + offset[i] * texel_size;
        sum += texture(tex, sample_uv) * kernel[i];
    }

    return sum;
}

const float StarWidth = 1.8;
const int Nstars = 1000;
const float noiselevel = 0.1;

void main() {
    //////////////////////////////////////////// 10
    int Npixels = int(clamp(10. * floor(f_time), 200, 200)); //100;

    vec2 sample_loc = floor(texcoord.xy * Npixels) / Npixels;
    fragColor = vec4(0.);

    for (int i = 0; i < Nstars; i++) {
        vec2 starpos = vec2(hash(i), hash(i+1));
        if (distance(sample_loc, starpos) < StarWidth / Npixels) {
            fragColor += 0.5 * pow(hash(i+2), 2.) * sqrt(hash(vec3(sample_loc.xy, f_time)));
        }
    }

    // add the asteroid
    if (distance(sample_loc, vec2(asteroid_pos[0], asteroid_pos[1])) < StarWidth / Npixels) {
        fragColor += asteroid_brightness;
    }

    fragColor += vec4(noiselevel * pow(hash(sample_loc.xy + vec2(floor(10.*f_time) / 10.)), 1.5));
    fragColor = clamp(fragColor, 0, 1);
    fragColor.w = 1.;
}