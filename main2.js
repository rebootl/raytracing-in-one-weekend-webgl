/* eslint no-console:0 consistent-return:0 */
"use strict";

function createShader(gl, type, source) {
  var shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  var success = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
  if (success) {
    return shader;
  }

  console.log(gl.getShaderInfoLog(shader));
  gl.deleteShader(shader);
}

function createProgram(gl, vertexShader, fragmentShader) {
  var program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  var success = gl.getProgramParameter(program, gl.LINK_STATUS);
  if (success) {
    return program;
  }

  console.log(gl.getProgramInfoLog(program));
  gl.deleteProgram(program);
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

  uniform vec2 canvasDimensions;

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
  };

  float hitSphere(const Sphere s, const Ray r, const float tMin,
      const float tMax) {
    vec3 oc = r.origin - s.center;
    float a = pow(length(r.direction), 2.);
    float halfB = dot(oc, r.direction);
    float c = pow(length(oc), 2.) - s.radius * s.radius;
    float discriminant = halfB * halfB - a * c;
    if (discriminant < 0.) {
      return -1.0;
    }
    return (-halfB - sqrt(discriminant) ) / a;
  }

  vec3 at(const float t, const Ray r) {
    return r.origin + t * r.direction;
  }

  vec3 rayColor(const Ray r, const Sphere[2] spheres) {
    for (int i = 0; i < 2; i++) {
      float t = hitSphere(spheres[1], r, 0., 99999.);
      if (t > 0.) {
        vec3 N = normalize(at(t, r) - vec3(0, 0, -1));
        return 0.5 * (N + 1.);
      }
      t = 0.5 * (normalize(r.direction).y + 1.);
      return (1. - t) * vec3(1., 1., 1.) + t * vec3(0.5, 0.7, 1.0);
    }
  }

  void main() {
    vec2 texcoord = gl_FragCoord.xy / canvasDimensions;
    // compute texcoord from gl_FragCoord;
    vec2 uv = gl_FragCoord.xy / (canvasDimensions - 1.);

    float aspectRatio = canvasDimensions.x / canvasDimensions.y;

    // Camera

    float viewportHeight = 2.0;
    float viewportWidth = aspectRatio * viewportHeight;
    float focalLength = 1.0;

    vec3 origin = vec3(0, 0, 0);
    vec3 horizontal = vec3(viewportWidth, 0, 0);
    vec3 vertical = vec3(0, viewportHeight, 0);
    vec3 lowerLeftCorner = origin - horizontal / 2. - vertical / 2.
      - vec3(0, 0, focalLength);

    Ray r = Ray(
      origin,
      lowerLeftCorner + uv.x * horizontal + uv.y * vertical - origin
    );

    Sphere spheres[2];
    spheres[0] = Sphere(vec3(0, 0, -1), 0.5);
    spheres[1] = Sphere(vec3(0, -100.5, -1), 100.);
    //spheres[1] = Sphere(vec3(0, 1, -1), 0.4);

    // gl_FragColor is a special variable a fragment shader
    // is responsible for setting
    gl_FragColor = vec4(rayColor(r, spheres), 1);
  }
`;

function main() {
  // Get A WebGL context
  var canvas = document.querySelector("#canvas");
  var gl = canvas.getContext("webgl");
  if (!gl) {
    return;
  }

  canvas.height = 400;

  // create GLSL shaders, upload the GLSL source, compile the shaders
  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vs);
  const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fs);

  // Link the two shaders into a program
  var program = createProgram(gl, vertexShader, fragmentShader);

  // look up where the vertex data needs to go.
  var positionAttributeLocation = gl.getAttribLocation(program, "a_position");

  const canvasDimensionsLocation = gl.getUniformLocation(program, 'canvasDimensions');

  // Create a buffer and put three 2d clip space points in it
  var positionBuffer = gl.createBuffer();

  // Bind it to ARRAY_BUFFER (think of it as ARRAY_BUFFER = positionBuffer)
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

  var positions = [
    -1, -1,
    -1, 1,
    1, -1,
    1, -1,
    -1, 1,
    1, 1
  ];
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

  // code above this line is initialization code.
  // code below this line is rendering code.

  webglUtils.resizeCanvasToDisplaySize(gl.canvas);

  // Tell WebGL how to convert from clip space to pixels
  gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

  // Clear the canvas
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);

  // Tell it to use our program (pair of shaders)
  gl.useProgram(program);

  // Turn on the attribute
  gl.enableVertexAttribArray(positionAttributeLocation);

  // Bind the position buffer.
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

  // Tell the attribute how to get data out of positionBuffer (ARRAY_BUFFER)
  var size = 2;          // 2 components per iteration
  var type = gl.FLOAT;   // the data is 32bit floats
  var normalize = false; // don't normalize the data
  var stride = 0;        // 0 = move forward size * sizeof(type) each iteration to get the next position
  var offset = 0;        // start at the beginning of the buffer
  gl.vertexAttribPointer(
      positionAttributeLocation, size, type, normalize, stride, offset);

  gl.uniform2f(canvasDimensionsLocation, gl.canvas.width, gl.canvas.height);

  // draw
  var primitiveType = gl.TRIANGLES;
  var offset = 0;
  var count = 6;
  gl.drawArrays(primitiveType, offset, count);
}

main();
