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
  precision highp float;

  uniform sampler2D inputTex;
  uniform vec2 canvasDimensions;
  uniform float time;
  uniform float niter;

  const int NSPHERES = 2;
  const int MAX_DEPTH = 15;

  struct Ray {
    vec3 origin;
    vec3 direction;
  };

  struct Sphere {
    vec3 center;
    float radius;
    // .. mat. etc.
  };

  struct hitRecord {
    vec3 p;
    vec3 normal;
    float t;
    bool frontFace;
  };

/*
  #define PHI 1.61803398874989484820459
  #define PI 3.1415926535897932385
  #define TAU 2. * PI

  float g_seed = 0.25;

  bool isnan(float x){
    return !(x > 0. || x < 0. || x == 0.);
  }

  float random(vec2 st) {
       return fract(tan(distance(st*PHI, st)*g_seed)*st.x);
  }

  vec2 random2(float seed) {
    return vec2(
      random(vec2(seed-1.23, (seed+3.1)* 3.2)),
      random(vec2(seed+12.678, seed - 5.8324))
      );
  }

  vec3 random3(float seed) {
    return vec3(
      random(vec2(seed-0.678, seed-0.123)),
      random(vec2(seed-0.3, seed+0.56)),
      random(vec2(seed+0.1234, seed-0.523))
      );
  }

  vec3 randomInUnitSphere(float seed) {
    vec2 tp = random2(seed);
    float theta = tp.x * TAU;
    float phi = tp.y * TAU;
    vec3 p = vec3(sin(theta) * cos(phi), sin(theta)*sin(phi), cos(theta));

    return normalize(p);
  }

  vec3 randomUnit(float seed){
      vec2 rand = random2(seed);
      float a = rand.x * TAU;
      float z = (2. * rand.y) - 1.;
      float r = sqrt(1. - z*z);
      return vec3(r*cos(a), r*sin(a), z);
  }
*/

  float _x;

  void seed(float s) {
    _x = s;
  }

  float randomLCG() {
    const float m = pow(2., 63.) - 1.;
    const float a = 48271.;
    _x = mod(a * _x, m);
    return _x;
  }

  float random(float min, float max) {
    float r = randomLCG() / (pow(2., 63.) - 1.);
    return min + (max - min) * r;
  }

  vec2 randomVec2(float min, float max) {
    return vec2(random(min, max), random(min, max));
  }

  vec3 randomVector(float min, float max) {
    return vec3(random(min, max), random(min, max), random(min, max));
  }

  vec3 randomInUnitSphere() {
    for (int s = 0; s < 10; s++) {
      vec3 p = randomVector(-1., 1.);
      if (pow(length(p), 2.) >= 1.) continue;
      return p;
    }
    // return anyways
    return randomVector(-1., 1.);
  }

  void setFaceNormal(const Ray r, const vec3 outwardNormal, inout hitRecord rec) {
    rec.frontFace = dot(r.direction, outwardNormal) < 0.;
    rec.normal = rec.frontFace ? outwardNormal : -outwardNormal;
  }

  vec3 at(const float t, const Ray r) {
    return r.origin + t * r.direction;
  }

  bool hitSphere(const Sphere s, const Ray r, const float tMin,
      const float tMax, out hitRecord rec) {
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
    return true;
  }

  bool worldHit(const Sphere[NSPHERES] spheres, const Ray r, const float tMin,
      const float tMax, out hitRecord rec) {
    hitRecord tempRec;
    bool hitAnything = false;
    float closestSoFar = tMax;

    for (int i = 0; i < NSPHERES; i++) {
      if (hitSphere(spheres[i], r, 0., closestSoFar, tempRec)) {
        hitAnything = true;
        closestSoFar = tempRec.t;
        rec = tempRec;
      }
    }
    return hitAnything;
  }

  vec3 rayColor(Ray r, const Sphere[NSPHERES] spheres) {

    hitRecord rec;

    // background
    float t = 0.5 * (normalize(r.direction).y + 1.);
    vec3 col = (1. - t) * vec3(1., 1., 1.) + t * vec3(0.5, 0.7, 1.0);

    for (int s = 0; s < 2; s++) {

      bool hit = worldHit(spheres, r, 0., 99999., rec);

      if (!hit) {
        return col;
      }
      vec3 jitter = randomInUnitSphere();
      /*if (isnan(jitter.r) || isnan(jitter.g) || isnan(jitter.b)) {
        jitter = vec3(0.);
      }*/
      vec3 target = rec.p + rec.normal + jitter;
      r = Ray(rec.p, target - rec.p);
      col = 0.5 * col;
    }

    return col;
  }

  void main() {
    vec2 uv = gl_FragCoord.xy / canvasDimensions;

    /*
    g_seed = random(gl_FragCoord.xy * (mod(time, 100.)));
    if(isnan(g_seed)){
      g_seed = 0.25;
    }
    */
    seed(gl_FragCoord.x * gl_FragCoord.y * time);

    float aspectRatio = canvasDimensions.x / canvasDimensions.y;

    // Camera

    const float viewportHeight = 2.0;
    float viewportWidth = aspectRatio * viewportHeight;
    const float focalLength = 1.0;

    const vec3 origin = vec3(0, 0, 0);
    vec3 horizontal = vec3(viewportWidth, 0, 0);
    vec3 vertical = vec3(0, viewportHeight, 0);
    vec3 lowerLeftCorner = origin - horizontal / 2. - vertical / 2.
      - vec3(0, 0, focalLength);

    Sphere spheres[NSPHERES];
    spheres[0] = Sphere(vec3(0, 0, -1), 0.5);
    spheres[1] = Sphere(vec3(0, -100.5, -1), 100.);
    //spheres[1] = Sphere(vec3(0, 1, -1), 0.4);

    //vec3 pixelColor = vec3(0, 0, 0);

    // anti aliasing
    vec2 jitter = (2. * randomVec2(0., 1.)) - 1.;
    vec2 st = uv;
    // check for NaN leakage
    /*
    if (isnan(st.x) || isnan(st.y)) {
      st = uv;
    }*/

    //float u = (gl_FragCoord.x + random2(g_seed).x) / (canvasDimensions.x - 1.);
    //float v = (gl_FragCoord.y + random2(g_seed).y) / (canvasDimensions.y - 1.);

    Ray r = Ray(
      origin,
      lowerLeftCorner + st.x * horizontal + st.y * vertical - origin
    );
    vec3 pixelColor = rayColor(r, spheres);
    //vec3 pixelColor = random3(g_seed);

    vec4 inputColor = texture2D(inputTex, uv);

    // sample + gamma corr.
    //float scale = 1. / float(SAMPLES_PPX);
    //pixelColor = (vec4(pixelColor, 1) + currentColor) / 2.;
    // gl_FragColor is a special variable a fragment shader
    // is responsible for setting
    gl_FragColor = vec4(pixelColor, 1) + inputColor;
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
  /*const texdata2 = new Float32Array(
    new Array(canvas.width * canvas.height).fill(0)
      .map(() => [0, 0, 0, 0]).flat()
  );*/
  const texdatar = new Float32Array(
    new Array(canvas.width * canvas.height * 4).fill(0)
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
  const niter = 1;
  // Clear the canvas
  //webglUtils.resizeCanvasToDisplaySize(gl.canvas);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);

  // Tell WebGL how to convert from clip space to pixels
  gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

  for (let i = 0; i < niter; i++) {

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
    gl.uniform1f(timeLocation, rand(0, 10));
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

    /*const t1 = outputFB1;
    outputFB1 = outputFB2;
    outputFB2 = t1;
    const t2 = outputTex1;
    outputTex1 = outputTex2;
    outputTex2 = t2;*/
  }
}

main();
