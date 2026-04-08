uniform vec3 sunDir;         // Sun direction in world space (normalized)
uniform vec3 galacticDir;    // Galactic plane normal (normalized)
in vec3 v_model_pos;;     // Direction from camera

// --- Utility random/noise ---
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453);
}

float noise(vec2 p){
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash(i);
    float b = hash(i+vec2(1.0,0.0));
    float c = hash(i+vec2(0.0,1.0));
    float d = hash(i+vec2(1.0,1.0));
    vec2 u = f*f*(3.0-2.0*f);
    return mix(a,b,u.x) + (c-a)*u.y*(1.0-u.x) + (d-b)*u.x*u.y;
}

// --- Base sky colors ---
vec3 skyColor(float alt) {
    vec3 nightColor = vec3(0.02, 0.04, 0.1);
    vec3 duskColor  = vec3(0.9, 0.4, 0.2);
    vec3 dayColor   = vec3(0.5, 0.7, 1.0);

    return (alt > 0.0) ? mix(duskColor, dayColor, alt)
                       : mix(duskColor, nightColor, -alt);
}

// --- Horizon brightness ---
float horizonBrightness(vec3 dir) {
    float y = clamp(dir.y, -1.0, 1.0);
    return 0.5 + 0.5 * exp(-pow(y * 3.0, 2.0));
}

// --- Painterly sun gradient ---
vec3 sunGradient(vec3 dir, vec3 baseColor, vec3 sunDir) {
    float alignment = max(dot(dir, sunDir), 0.0);
    float layer1 = pow(alignment, 64.0);
    float layer2 = pow(alignment, 16.0);
    float layer3 = pow(alignment, 4.0);

    vec3 coreColor = vec3(1.0, 0.95, 0.8);
    vec3 innerHalo = vec3(1.0, 0.6, 0.4);
    vec3 outerGlow = vec3(0.8, 0.5, 0.7);

    vec3 sunCol = coreColor * layer1 + innerHalo * layer2 + outerGlow * layer3;
    return mix(baseColor, baseColor + sunCol, clamp(alignment, 0.0, 1.0));
}

// --- Stars ---
vec3 stars(vec3 dir) {
    float theta = acos(dir.y);
    float phi = atan(dir.z, dir.x);
    vec2 seed = vec2(phi/3.14159, theta/3.14159);
    float brightness = pow(hash(seed*1000.0), 40.0);
    return vec3(brightness);
}

// --- Milky Way band aligned with galacticDir ---
vec3 milkyWay(vec3 dir, vec3 galacticDir) {
    // Distance to galactic plane (0 at plane, ±1 at poles)
    float band = abs(dot(dir, galacticDir));

    // Band glow (strongest near galactic plane)
    float glow = exp(-pow(band*6.0,2.0));

    // Noise modulation for dust and clouds
    float n1 = noise(dir.xz*40.0);
    float n2 = noise(dir.xz*80.0 + 10.0);
    float dust = smoothstep(0.3, 0.7, n1*0.6+n2*0.4);

    // Colors
    vec3 base = vec3(0.6,0.65,0.9);   // bluish-white starlight
    vec3 nebula = vec3(0.8,0.4,0.6);  // magenta nebula tint
    vec3 dustLane = vec3(0.05,0.05,0.08); // dark dust lanes

    vec3 col = mix(base, nebula, n1*0.5);
    col = mix(col, dustLane, dust*0.8);
    col *= glow*1.5;

    return col;
}

void main() {
    vec3 dir = normalize(v_model_pos);

    float sun_alt = normalize(sunDir).y;

    // Base sky
    vec3 col = skyColor(sun_alt);
    col *= horizonBrightness(dir);

    // Sun
    col = sunGradient(dir, col, normalize(sunDir));

    // Night fade factor
    float nightFactor = clamp(-sun_alt, 0.0, 1.0);

    // Stars + Milky Way
    col += stars(dir) * nightFactor;
    col += milkyWay(dir, normalize(galacticDir)) * nightFactor;

    gl_FragColor = vec4(col, 1.0);
}
