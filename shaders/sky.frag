#version 430
//uniform sampler2D p3d_Texture0; // first texture unit
in vec3 v_model_pos;
uniform float u_time;
uniform float sun_dec;
out vec4 fragColor;

vec3 rotateAroundAxis(vec3 v, vec3 axis, float angle) {
    axis = normalize(axis);
    float c = cos(angle);
    float s = sin(angle);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

float remap(float v, float from_in, float to_in, float from_out, float to_out) {
    return from_out + (v - from_in) * (to_out - from_out) / (to_in - from_in);
}

// 3D hash
float hash(vec3 p) {
    return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453);
}

// Smooth interpolation
vec3 fade(vec3 t) {
    return t * t * (3.0 - 2.0 * t);
}

// Value noise in 3D
float valuenoise3(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);

    // 8 corners of cube
    float n000 = hash(i + vec3(0.0, 0.0, 0.0));
    float n100 = hash(i + vec3(1.0, 0.0, 0.0));
    float n010 = hash(i + vec3(0.0, 1.0, 0.0));
    float n110 = hash(i + vec3(1.0, 1.0, 0.0));
    float n001 = hash(i + vec3(0.0, 0.0, 1.0));
    float n101 = hash(i + vec3(1.0, 0.0, 1.0));
    float n011 = hash(i + vec3(0.0, 1.0, 1.0));
    float n111 = hash(i + vec3(1.0, 1.0, 1.0));

    // Fade curves
    vec3 u = fade(f);

    // Trilinear interpolation
    return mix(
        mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
        mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
        u.z
    );
}




void main() {

    const float PI = 3.14159265359;
    float lat = 52.; // in degrees

    vec3 sun_pos = vec3(
        cos(-2*PI * u_time) * cos(radians(sun_dec)),
        sin(radians(sun_dec)),
        sin(-2*PI * u_time) * cos(radians(sun_dec))
    );
    sun_pos = rotateAroundAxis(sun_pos, vec3(0.,0.,1.), radians(90-lat));
    vec3 rot_sky = rotateAroundAxis(v_model_pos, vec3(-cos(radians(lat)), sin(radians(lat)), 0), -2*PI * u_time);

    float atm_weight = pow(normalize(v_model_pos).y, 0.5);

    // linear interpolation of the bottom/top colors of the sky gradient
    vec3 bottom_color = vec3(0,0,0);
    vec3 top_color = vec3(1,1,1);
    float stellar_brightness = 0;
    if (sun_pos.y >= 0.1) {
        bottom_color = vec3(0.6706, 0.749, 0.9137);
        top_color = vec3(0.2431, 0.1647, 0.5882);
    } else {
        if (sun_pos.y > -0.1) {
            bottom_color = mix(
                vec3(0.6706, 0.749, 0.9137),
                vec3(0.8392, 0.5412, 0.3412),
                remap(sun_pos.y, 0.1, -0.1, 0., 1.)
            );
            top_color = mix(
                vec3(0.2431, 0.1647, 0.5882),
                vec3(0.0588, 0.0353, 0.1412),
                remap(sun_pos.y, 0.1, -0.1, 0., 1.)
            );
        } else {
            if (sun_pos.y > -0.2) {
                bottom_color = mix(
                    vec3(0.8392, 0.5412, 0.3412),
                    vec3(0.0667, 0.0667, 0.0706),
                    remap(sun_pos.y, -0.1, -0.2, 0., 1.)
                );
                top_color = mix(
                    vec3(0.0588, 0.0353, 0.1412),
                    vec3(0.0157, 0.0039, 0.051),
                    remap(sun_pos.y, -0.1, -0.2, 0., 1.)
                );
                stellar_brightness = clamp(remap(sun_pos.y, -0.1, -0.2, 0., 1.), 0, 1);
            } else {
                bottom_color = vec3(0.0667, 0.0667, 0.0706);
                top_color = vec3(0.0157, 0.0039, 0.051);
                stellar_brightness = 1;
            }
        }
    }

    // mix the two skyboxes
    vec3 sky_color = mix(
        bottom_color,
        top_color,
        clamp(atm_weight, 0, 1)
    );



    float sun_ang_dist = acos(dot(sun_pos, normalize(v_model_pos)));
    if (sun_ang_dist <= radians(0.5)) {
        sky_color += vec3(1., 1., 1.);
    }

    // add sun bloom
    sky_color += vec3(exp(-sun_ang_dist/0.05));
    sky_color += 0.5 * exp(-sun_ang_dist/0.5) * vec3(0.8745, 0.5176, 0.1373) * atm_weight;

    // add stars
    sky_color += stellar_brightness * vec3(pow(clamp(remap(valuenoise3(5*rot_sky), 0, 1, -5, 1), 0, 1), 3)) * atm_weight;

    // add the milky way
    sky_color += stellar_brightness * vec3(0.2*exp(-pow(dot(normalize(rot_sky), normalize(vec3(6, 67, 15))), 2)/0.05));

    if (v_model_pos.y < 0) {
        sky_color = vec3(0.);
    }

    fragColor = vec4(
        sky_color, 1.
    );
}
