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
precision mediump float;

uniform vec2 canvasDimensions;
uniform float time;

//uniform int	PASSINDEX;
uniform sampler2D inputTex;
//uniform sampler2D view;
//uniform sampler2D backbuffer;

#define MAX_FLOAT 1e5
#define MAX_RECURSION 5
#define PI 3.1415926535897932385
#define TAU 2. * PI
// Φ = Golden Ratio
#define PHI 1.61803398874989484820459


float g_seed = 0.25;

float deg2rad(float deg){
  return deg*PI / 180.;
}

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


// ray primitive

struct ray{
  vec3 o, dir;
};

vec3 ray_at(ray r, float t){
  return r.o + r.dir*t;
}

struct hit{
  vec3 n,p;
  float t;
  bool front_face;
};


// camera

struct camera{
  vec3 o, lower_left, horizontal, vertical;
};

camera make_camera(){
  float h = 2.0;
  float aspect = (canvasDimensions.x / canvasDimensions.y);
  float w = h * aspect;

  vec3 o = vec3(0.,0.,0.);
  vec3 horizontal = vec3(w, 0,0);
  vec3 vertical = vec3(0.,h,0.);
  vec3 lower_left = o - horizontal / 2. - vertical / 2. -
    vec3(0, 0, 1.);
  return camera(o, lower_left, horizontal, vertical);
}

ray camera_get_ray(camera c, vec2 uv){
  ray r =  ray(c.o,
    normalize(c.lower_left + uv.x * c.horizontal + uv.y * c.vertical - c.o));
    return r;
}


// intersections

struct sphere{
  vec3 center;
  float radius;
};

bool hit_sphere(sphere s, ray r, vec2 t_min_max, inout hit rec){
  vec3 oc = r.o - s.center;
  float b = dot(oc, r.dir);
  float c = dot(oc,oc) - s.radius*s.radius;
  float d = b*b - c;
  if(d < 0.){ return false; }

  float t1 = (-b - sqrt(d));
  float t2 = (-b + sqrt(d));
  float t = t1 < t_min_max.x ? t2 : t1;

  vec3 p = ray_at(r, t);
  vec3 n = (p - s.center);
  // if front_face, the ray is in the sphere and not out, so invert normal
  bool front_face = dot(r.dir, n) > 0.;

  n = front_face ? -n : n;
  n /= s.radius;
  if(t < t_min_max.x || t > t_min_max.y){
    return false;
  }

  rec = hit(n,p,t, front_face);
  return true;
}


// raytracing

bool raycast(const in ray r, vec2 t_min_max, inout hit h){
  sphere s = sphere(vec3(0,0,-2.), 1.);
  sphere pl = sphere(vec3(0.,-201, -2.), 200.);

  bool is_hit = false;
  is_hit = hit_sphere(pl, r, t_min_max, h) || is_hit;
  is_hit = hit_sphere(s, r, t_min_max, h) || is_hit;
  return is_hit;
}

vec3 ray_color(ray r){
  vec2 t_min_max = vec2(0.001, MAX_FLOAT);
  hit h;
  vec3 col = vec3(1.);

// #define debug_normal
#ifdef debug_normal
  if(raycast(r, t_min_max, h)){
    return .5 * (h.n + vec3(1.));
  }
#else

  for(int i=0; i < MAX_RECURSION; i++){
    bool is_hit = raycast(r, t_min_max, h);
    if(is_hit){
      vec3 jitter = random_unit(g_seed);
      if(isnan(jitter.r) || isnan(jitter.g) || isnan(jitter.b)){
        jitter = vec3(0.);
      }

      col *= 0.5;
      vec3 target = (h.p + h.n + jitter);
      r.o = h.p;
      r.dir = normalize(target - h.p);
    }else{

      vec3 unit_dir = normalize(r.dir);
      float t = 0.5 * (unit_dir.y + 1.0);
      col *= mix(vec3(1.0), vec3(0.5,0.7,1.0), t);
      return col;
    }
  }
  return col;
#endif
}

void main(){
  //vec2 p = (gl_FragCoord.xy * 2. - canvasDimensions) / min(canvasDimensions.x, canvasDimensions.y);
  vec2 uv = gl_FragCoord.xy / canvasDimensions;

  camera c = make_camera();
  // new seed every frame
  g_seed = random(gl_FragCoord.xy * (mod(time, 100.)));
  if(isnan(g_seed)){
    g_seed = 0.25;
  }

  // anti aliasing
  vec2 jitter = (2. * random2(g_seed)) - 1.;
  vec2 st = uv + jitter * 0.001;
  // check for NaN leakage
  if(isnan(st.x) || isnan(st.y)){
    st = uv;
  }

  // multisampling in veda
  // we use multipass, which does one pass (0) where we accumulate to a buffer (accum)
  // the second pass (1) takes the average of the accumulated values
  //if(PASSINDEX == 0){
  ray r = camera_get_ray(c, st);

  vec3 col = ray_color(r);

    // approximate gamma correction
    // col = sqrt(col);

  vec4 result = vec4(col,1.) + texture2D(inputTex, uv);

  if(result.a > MAX_FLOAT){
    result = vec4(result.xyz/result.a, 1.);
  }
  gl_FragColor = result;

/*
  if(PASSINDEX == 1){
    vec4 result = texture2D(accum, uv);

    gl_FragColor = vec4(result.xyz / result.a, 1.);
  }*/
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
  //const randTexLocation = gl.getUniformLocation(program, 'randTex');
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
  const niter = 250;
  // Clear the canvas
  //webglUtils.resizeCanvasToDisplaySize(gl.canvas);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);

  // Tell WebGL how to convert from clip space to pixels
  gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

  /*const texdatarand = new Float32Array(
    new Array(canvas.width * canvas.height * 4).fill(0)
      .map(() => [ rand(0, 1), 0, 0, 0 ]).flat()
  );
  const texr = createTexture(gl, texdatarand, canvas.width, canvas.height);
*/
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

    //gl.activeTexture(gl.TEXTURE0 + 1);
    //gl.bindTexture(gl.TEXTURE_2D, texr);

    gl.uniform1i(inputTexLocation, 0);
    gl.uniform2f(canvasDimensionsLocation, gl.canvas.width, gl.canvas.height);
    const t = Date.now() / 1000;
    const i = parseInt((t - parseInt(t)) * 1000);
    //console.log(i)
    gl.uniform1f(timeLocation, i);
    gl.uniform1f(niterLocation, niter);

    //gl.uniform1i(randTexLocation, 1);
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
