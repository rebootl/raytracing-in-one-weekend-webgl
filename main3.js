/* eslint no-console:0 consistent-return:0 */
"use strict";

const rand = (min, max) => {
  if (max === undefined) {
    max = min;
    min = 0;
  }
  return Math.random() * (max - min) + min;
};

function createShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  const success = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
  if (success) {
    return shader;
  }

  console.log(gl.getShaderInfoLog(shader));
  gl.deleteShader(shader);
}

function createProgram(gl, vertexShader, fragmentShader) {
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  const success = gl.getProgramParameter(program, gl.LINK_STATUS);
  if (success) {
    return program;
  }

  console.log(gl.getProgramInfoLog(program));
  gl.deleteProgram(program);
}

function createTexture(gl, data, width, height) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(
      gl.TEXTURE_2D,
      0,        // mip level
      gl.RGBA,  // internal format
      width,
      height,
      0,        // border
      gl.RGBA,  // format
      gl.FLOAT, // type
      data,
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return tex;
}

function createFramebuffer(gl, tex) {
  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  return fb;
}

const vs = `
  // an attribute will receive data from a buffer
  attribute vec4 a_position;

  // all shaders have a main function
  void main() {

    // gl_Position is a special variable a vertex shader
    // is responsible for setting
    gl_Position = a_position;
  }
`;

const fs = `
  // fragment shaders don't have a default precision so we need
  // to pick one. mediump is a good default
  precision mediump float;

  uniform sampler2D inputTex;
  uniform vec2 canvasDimensions;
  uniform float time;

  const int NSPHERES = 4;
  const int MAX_DEPTH = 25;


  #define MAX_FLOAT 1e5
  #define MAX_RECURSION 5
  #define PI 3.1415926535897932385
  #define TAU 2. * PI
  // Φ = Golden Ratio
  #define PHI 1.61803398874989484820459


  float deg2rad(float d) {
    return PI * d / 180.0;
  }

  float g_seed = 0.25;

  // random number generator

  //https://stackoverflow.com/a/34276128
  bool isnan(float x){
    return !(x > 0. || x < 0. || x == 0.);
  }

  // a variation of gold noise is used
  // https://stackoverflow.com/a/28095165
  // https://www.shadertoy.com/view/ltB3zD
  // centered around [0-1] in gaussian
  float random (vec2 st) {
       return fract(tan(distance(st*PHI, st)*g_seed)*st.x);
  }

  vec2 random2(float seed){
    return vec2(
      random(vec2(seed-1.23, (seed+3.1)* 3.2)),
      random(vec2(seed+12.678, seed - 5.8324))
      );
  }

  vec3 random3(float seed){
    return vec3(
      random(vec2(seed-0.678, seed-0.123)),
      random(vec2(seed-0.3, seed+0.56)),
      random(vec2(seed+0.1234, seed-0.523))
      );
  }

  vec3 random_in_unit_sphere( float seed) {
    vec2 tp = random2(seed);
    float theta = tp.x * TAU;
    float phi = tp.y * TAU;
    vec3 p = vec3(sin(theta) * cos(phi), sin(theta)*sin(phi), cos(theta));

    return normalize(p);
  }

  vec3 random_unit(float seed){
      vec2 rand = random2(seed);
      float a = rand.x * TAU;
      float z = (2. * rand.y) - 1.;
      float r = sqrt(1. - z*z);
      return vec3(r*cos(a), r*sin(a), z);
  }


  struct Ray {
    vec3 origin;
    vec3 direction;
  };

  struct Material {
    int type;
    vec3 color;
    float fuzz;
  };

  struct Sphere {
    vec3 center;
    float radius;
    Material mat;
  };

  struct hitRecord {
    vec3 p;
    vec3 normal;
    float t;
    bool frontFace;
    Material m;
  };

  bool nearZero(vec3 v) {
    float s = 1e-8;
    return (abs(v.x) < s && abs(v.y) < s && abs(v.z) < s);
  }

  vec3 jitter() {
    vec3 j = random_unit(g_seed);
    if(isnan(j.r) || isnan(j.g) || isnan(j.b)){
      j = vec3(0.);
    }
    return j;
  }

  vec3 diffuseMat(inout Ray r, hitRecord rec, vec3 col) {
    vec3 scatterDirection = rec.normal + jitter();
    if (nearZero(scatterDirection)) {
      scatterDirection = rec.normal;
    }

    vec3 target = rec.p + scatterDirection;
    r.origin = rec.p;
    //r.direction = normalize(target - rec.p);
    r.direction = scatterDirection;

    col = col * rec.m.color;
    return col;
  }

  vec3 metalMat(inout Ray r, hitRecord rec, vec3 col) {
    vec3 reflected = reflect(normalize(r.direction), rec.normal);
    r.origin = rec.p;
    r.direction = reflected + jitter() * rec.m.fuzz;
    col = col * rec.m.color;
    return col;
  }

  float reflectance(float cosine, float ref_idx) {
      // Use Schlick's approximation for reflectance.
      float r0 = (1. - ref_idx) / (1. + ref_idx);
      r0 = r0 * r0;
      return r0 + (1. - r0) * pow((1. - cosine), 5.);
  }

  vec3 refractMat(inout Ray r, hitRecord rec, vec3 col) {
    float ir = 1.5;
    float rratio = rec.frontFace ? (1.0/ir) : ir;

    vec3 unitDir = normalize(r.direction);
    float cos_theta = min(dot(-unitDir, rec.normal), 1.0);
    float sin_theta = sqrt(1.0 - cos_theta*cos_theta);

    bool cannotRefract = (rratio * sin_theta > 1.0);
    vec3 direction;

    if (cannotRefract || reflectance(cos_theta, rratio) > random2(g_seed).x)
        direction = reflect(unitDir, rec.normal);
    else
        direction = refract(unitDir, rec.normal, rratio);

    r.origin = rec.p;
    r.direction = direction;
    col = col * vec3(1., 1., 1.);
    return col;
  }

  void setFaceNormal(const Ray r, const vec3 outwardNormal, inout hitRecord rec) {
    rec.frontFace = dot(r.direction, outwardNormal) < 0.;
    rec.normal = rec.frontFace ? outwardNormal : -outwardNormal;
  }

  vec3 at(const float t, const Ray r) {
    return r.origin + t * r.direction;
  }

  bool hitSphere(const Sphere s, const Ray r, const float tMin,
      const float tMax, inout hitRecord rec) {
    vec3 oc = r.origin - s.center;
    float a = pow(length(r.direction), 2.);
    float halfB = dot(oc, r.direction);
    float c = pow(length(oc), 2.) - s.radius * s.radius;
    float discriminant = halfB * halfB - a * c;
    if (discriminant < 0.) {
      return false;
    }
    float sqrtd = sqrt(discriminant);

    float root = (-halfB - sqrtd) / a;
    if (root < tMin || tMax < root) {
      root = (-halfB + sqrtd) / a;
      if (root < tMin || tMax < root) {
        return false;
      }
    }
    rec.t = root;
    rec.p = at(root, r);
    vec3 outwardNormal = (rec.p - s.center) / s.radius;
    setFaceNormal(r, outwardNormal, rec);

    rec.m = s.mat;
    return true;
  }

  bool worldHit(const Sphere[NSPHERES] spheres, const Ray r, const float tMin,
      const float tMax, out hitRecord rec) {
    hitRecord tempRec;
    bool hitAnything = false;
    float closestSoFar = tMax;

    for (int i = 0; i < NSPHERES; i++) {
      if (hitSphere(spheres[i], r, tMin, closestSoFar, tempRec)) {
        hitAnything = true;
        closestSoFar = tempRec.t;
        rec = tempRec;
      }
    }
    return hitAnything;
  }

  vec3 rayColor(Ray r, const Sphere[NSPHERES] spheres) {

    hitRecord rec;

    vec3 col = vec3(1.);

    for (int s = 0; s < MAX_DEPTH; s++) {

      bool hit = worldHit(spheres, r, 0.001, MAX_FLOAT, rec);

      if (hit) {

        if (rec.m.type == 0) {
          col = diffuseMat(r, rec, col);
        }
        if (rec.m.type == 1) {
          col = metalMat(r, rec, col);
        }
        if (rec.m.type == 2) {
          col = refractMat(r, rec, col);
        }

        // normal
        //col = 0.5 * (rec.normal + vec3(1,1,1));
      } else {
        // background
        /*float t = 0.5 * (normalize(r.direction).y + 1.);
        vec3 col = (1. - t) * vec3(1., 1., 1.) + t * vec3(0.5, 0.7, 1.0);
        return col;
        */
        float t = 0.5 * (normalize(r.direction).y + 1.0);
        col *= mix(vec3(1.0), vec3(0.5,0.7,1.0), t);
        return col;
      }
    }
    return col;
  }

  void main() {
    vec2 uv = gl_FragCoord.xy / canvasDimensions;

    // new seed every frame
    g_seed = random(gl_FragCoord.xy * (mod(time, 100.)));
    if(isnan(g_seed)){
      g_seed = 0.25;
    }

    float aspectRatio = canvasDimensions.x / canvasDimensions.y;
    float vfov = 90.;

    vec3 lookFrom = vec3(-2, 2, 1);
    vec3 lookAt = vec3(0, 0, -1);
    vec3 vUp = vec3(0, 1, 0);

    // Camera

    float theta = deg2rad(vfov);
    float h = tan(theta/2.);

    float viewportHeight = 2.0 * h;
    float viewportWidth = aspectRatio * viewportHeight;
    const float focalLength = 1.0;

    vec3 w = normalize(lookFrom - lookAt);
    vec3 u = normalize(cross(vUp, w));
    vec3 v = cross(w, u);

    vec3 origin = lookFrom;
    vec3 horizontal = viewportWidth * u;
    vec3 vertical = viewportHeight * v;
    vec3 lowerLeftCorner = origin - horizontal / 2. - vertical / 2.
      - w;

    Material m1 = Material(0, vec3(0.7, 0.3, 0.3), 1.);
    Material m2 = Material(0, vec3(0.8, 0.8, 0.0), 1.);
    Material m3 = Material(2, vec3(0.8, 0.8, 0.8), 0.);
    Material m4 = Material(1, vec3(0.8, 0.6, 0.2), 0.8);

    Sphere spheres[NSPHERES];
    spheres[0] = Sphere(vec3(0, 0, -1), 0.5, m1);
    spheres[1] = Sphere(vec3(0, -100.5, -1), 100., m2);
    spheres[2] = Sphere(vec3(-1, 0, -1), 0.5, m3);
    spheres[3] = Sphere(vec3(1, 0, -1), 0.5, m4);

    // anti aliasing
    vec2 jitter = (2. * random2(g_seed)) - 1.;
    vec2 st = uv + jitter * 0.001;
    // check for NaN leakage
    if(isnan(st.x) || isnan(st.y)){
      st = uv;
    }

    Ray r = Ray(
      origin,
      normalize(lowerLeftCorner + st.x * horizontal + st.y * vertical - origin)
    );
    vec3 pixelColor = rayColor(r, spheres);

    vec4 result = vec4(pixelColor,1.) + texture2D(inputTex, uv);

    if(result.a > MAX_FLOAT){
      result = vec4(result.xyz/result.a, 1.);
    }
    gl_FragColor = result;
  }
`;

const drawVS = `
  // an attribute will receive data from a buffer
  attribute vec4 a_position;

  // all shaders have a main function
  void main() {

    // gl_Position is a special variable a vertex shader
    // is responsible for setting
    gl_Position = a_position;
  }
`;

const drawFS = `
  precision highp float;

  uniform sampler2D outputTex;
  uniform vec2 canvasDimensions;
  uniform float niter;

  void main() {
    vec4 pixelColor = texture2D(outputTex, gl_FragCoord.xy / canvasDimensions);
    gl_FragColor = sqrt(pixelColor / niter);
  }
`;

function main() {
  // Get A WebGL context
  const canvas = document.querySelector("#canvas");
  const gl = canvas.getContext("webgl");
  if (!gl) {
    return;
  }

  // check we can use floating point textures
  const ext1 = gl.getExtension('OES_texture_float');
  if (!ext1) {
    alert('Need OES_texture_float');
    return;
  }
  // check we can render to floating point textures
  const ext2 = gl.getExtension('WEBGL_color_buffer_float');
  if (!ext2) {
    alert('Need WEBGL_color_buffer_float');
    return;
  }
  // check we can use textures in a vertex shader
  if (gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS) < 1) {
    alert('Can not use textures in vertex shaders');
    return;
  }

  canvas.height = 400;
  canvas.width = 600;

  //console.log(canvas.height * canvas.width)

  // create GLSL shaders, upload the GLSL source, compile the shaders
  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vs);
  const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fs);

  const drawVertexShader = createShader(gl, gl.VERTEX_SHADER, drawVS);
  const drawFragmentShader = createShader(gl, gl.FRAGMENT_SHADER, drawFS);

  // Link the two shaders into a program
  const program = createProgram(gl, vertexShader, fragmentShader);
  const drawProgram = createProgram(gl, drawVertexShader, drawFragmentShader);

  // look up where the vertex data needs to go.
  const positionAttributeLocation = gl.getAttribLocation(program, 'a_position');
  const canvasDimensionsLocation = gl.getUniformLocation(program, 'canvasDimensions');
  const inputTexLocation = gl.getUniformLocation(program, 'inputTex');
  const timeLocation = gl.getUniformLocation(program, 'time');
  const niterLocation = gl.getUniformLocation(program, 'niter');

  const drawPositionAttributeLocation = gl.getAttribLocation(drawProgram, 'a_position');
  const drawCanvasDimensionsLocation = gl.getUniformLocation(drawProgram, 'canvasDimensions');
  const drawOutputTexLocation = gl.getUniformLocation(drawProgram, 'outputTex');
  const drawNiterLocation = gl.getUniformLocation(drawProgram, 'niter');

  // Create a buffer and put three 2d clip space points in it
  const positionBuffer = gl.createBuffer();

  // Bind it to ARRAY_BUFFER (think of it as ARRAY_BUFFER = positionBuffer)
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

  const positions = [
    -1, -1,
    -1, 1,
    1, -1,
    1, -1,
    -1, 1,
    1, 1
  ];
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

  const texdata = new Float32Array(
    new Array(canvas.width * canvas.height).fill(0)
      .map(() => [0, 0, 0, 0]).flat()
  );

  const tex1 = createTexture(gl, texdata, canvas.width, canvas.height);
  const tex2 = createTexture(gl, texdata, canvas.width, canvas.height);

  let input = {
    tex: tex1,
    FB: createFramebuffer(gl, tex1)
  };
  let output = {
    tex: tex2,
    FB: createFramebuffer(gl, tex2)
  };

  // code above this line is initialization code.
  // code below this line is rendering code.
  let niter = 0;
  // Clear the canvas
  //webglUtils.resizeCanvasToDisplaySize(gl.canvas);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);

  // Tell WebGL how to convert from clip space to pixels
  gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

  //for (let i = 0; i < niter; i++) {
  function render() {

    gl.bindFramebuffer(gl.FRAMEBUFFER, output.FB);
    // Tell it to use our program (pair of shaders)
    gl.useProgram(program);

    // Turn on the attribute
    gl.enableVertexAttribArray(drawPositionAttributeLocation);

    // Bind the position buffer.
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

    // Tell the attribute how to get data out of positionBuffer (ARRAY_BUFFER)
    //const size = 2;          // 2 components per iteration
    //const type = gl.FLOAT;   // the data is 32bit floats
    //const normalize = false; // don't normalize the data
    //const stride = 0;        // 0 = move forward size * sizeof(type) each iteration to get the next position
    //const offset = 0;        // start at the beginning of the buffer
    gl.vertexAttribPointer(drawPositionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, input.tex);

    gl.uniform1i(inputTexLocation, 0);
    gl.uniform2f(canvasDimensionsLocation, gl.canvas.width, gl.canvas.height);
    const t = Date.now() / 1000;
    const i = parseInt((t - parseInt(t)) * 1000);
    gl.uniform1f(timeLocation, i);
    gl.uniform1f(niterLocation, niter);

    // draw
    // primitiveType, offset, count
    gl.drawArrays(gl.TRIANGLES, 0, 6);



    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    // Tell WebGL how to convert from clip space to pixels
    //gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    // Clear the canvas
    //gl.clearColor(0, 0, 0, 0);
    //gl.clear(gl.COLOR_BUFFER_BIT);

    // Tell it to use our program (pair of shaders)
    gl.useProgram(drawProgram);

    // Turn on the attribute
    gl.enableVertexAttribArray(positionAttributeLocation);

    // Bind the position buffer.
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

    // Tell the attribute how to get data out of positionBuffer (ARRAY_BUFFER)
    gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, output.tex);

    gl.uniform1i(drawOutputTexLocation, 0);
    gl.uniform2f(drawCanvasDimensionsLocation, gl.canvas.width, gl.canvas.height);
    gl.uniform1f(drawNiterLocation, niter);

    // draw
    // primitiveType, offset, count
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    const t1 = output;
    output = input;
    input = t1;

    const n = document.querySelector('.niter');
    n.textContent = niter;
    niter++;
    if (run) {
      requestAnimationFrame(render);
    }
  }
  return render;
}

let run = true;

function reset() {
  run = false;
  const r = main();
  run = true;
  requestAnimationFrame(r);
  return r;
}

let r = reset();

function toggle() {
  if (run) {
    run = false;
  } else {
    run = true;
    requestAnimationFrame(r);
  }
}
