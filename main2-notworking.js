'use strict';

/* eslint no-alert: 0 */

const rand = (min, max) => {
  if (max === undefined) {
    max = min;
    min = 0;
  }
  return Math.random() * (max - min) + min;
};

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

const updateVelocityVS = `
  attribute vec4 velocity;
  void main() {
    gl_Position = velocity;
  }
`;

const updateVelocityFS = `
  precision highp float;

  uniform sampler2D velocityTex;
  uniform vec2 texDimensions;
  uniform float deltaTime;

  void main() {
    // compute texcoord from gl_FragCoord;
    vec2 texcoord = gl_FragCoord.xy / texDimensions;

    vec2 v = texture2D(velocityTex, texcoord).xy;

    gl_FragColor = vec4(v.x, v.y, 0, 1);
  }
`;

const updatePositionVS = `
  attribute vec4 position;
  void main() {
    gl_Position = position;
  }
`;

const updatePositionFS = `
  precision highp float;

  uniform sampler2D positionTex;
  uniform sampler2D velocityTex;
  uniform vec2 texDimensions;
  uniform vec2 canvasDimensions;
  uniform float deltaTime;

  vec2 euclideanModulo(vec2 n, vec2 m) {
    return mod(mod(n, m) + m, m);
  }

  void main() {
    // there will be one velocity per position
    // so the velocity texture and position texture
    // are the same size.

    // further, we're generating new positions
    // so we know our destination is the same size
    // as our source so we only need one set of
    // shared texture dimensions

    // compute texcoord from gl_FragCoord;
    vec2 texcoord = gl_FragCoord.xy / texDimensions;

    vec2 p = texture2D(positionTex, texcoord).xy;
    vec2 v = texture2D(velocityTex, texcoord).xy;

    vec2 newPosition = euclideanModulo(vec2(
      p.x + v.x * deltaTime,
      p.y + v.y * deltaTime
    ), canvasDimensions);

    gl_FragColor = vec4(newPosition, 0, 1);
  }
`;

const drawParticlesVS = `
  attribute float id;
  uniform sampler2D positionTex;
  uniform vec2 texDimensions;
  uniform mat4 matrix;

  vec4 getValueFrom2DTextureAs1DArray(sampler2D tex, vec2 dimensions, float index) {
    float y = floor(index / dimensions.x);
    float x = mod(index, dimensions.x);
    vec2 texcoord = (vec2(x, y) + 0.5) / dimensions;
    return texture2D(tex, texcoord);
  }

  void main() {
    // pull the position from the texture
    vec4 position = getValueFrom2DTextureAs1DArray(positionTex, texDimensions, id);

    // do the common matrix math
    gl_Position = matrix * vec4(position.xy, 0, 1);
    gl_PointSize = 5.0;
  }
`;

const drawParticlesFS = `
  precision highp float;
  void main() {
    gl_FragColor = vec4(0.4, 0.7, 0.6, 1);
  }
`;

function main() {

  // Get A WebGL context
  /** @type {HTMLCanvasElement} */
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

  const updateVelocityProgram = webglUtils.createProgramFromSources(
      gl, [updateVelocityVS, updateVelocityFS]);
  const updatePositionProgram = webglUtils.createProgramFromSources(
      gl, [updatePositionVS, updatePositionFS]);
  const drawParticlesProgram = webglUtils.createProgramFromSources(
      gl, [drawParticlesVS, drawParticlesFS]);

  const updateVelocityPrgLocs = {
    velocity: gl.getAttribLocation(updateVelocityProgram, 'velocity'),
    velocityTex: gl.getUniformLocation(updateVelocityProgram, 'velocityTex'),
    texDimensions: gl.getUniformLocation(updateVelocityProgram, 'texDimensions'),
    deltaTime: gl.getUniformLocation(updateVelocityProgram, 'deltaTime'),
  };

  const updatePositionPrgLocs = {
    position: gl.getAttribLocation(updatePositionProgram, 'position'),
    positionTex: gl.getUniformLocation(updatePositionProgram, 'positionTex'),
    velocityTex: gl.getUniformLocation(updatePositionProgram, 'velocityTex'),
    texDimensions: gl.getUniformLocation(updatePositionProgram, 'texDimensions'),
    canvasDimensions: gl.getUniformLocation(updatePositionProgram, 'canvasDimensions'),
    deltaTime: gl.getUniformLocation(updatePositionProgram, 'deltaTime'),
  };

  const drawParticlesProgLocs = {
    id: gl.getAttribLocation(drawParticlesProgram, 'id'),
    positionTex: gl.getUniformLocation(drawParticlesProgram, 'positionTex'),
    texDimensions: gl.getUniformLocation(drawParticlesProgram, 'texDimensions'),
    matrix: gl.getUniformLocation(drawParticlesProgram, 'matrix'),
  };

  // setup a full canvas clip space quad
  const updateVelocitiesBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, updateVelocitiesBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1,
     1, -1,
    -1,  1,
    -1,  1,
     1, -1,
     1,  1,
  ]), gl.STATIC_DRAW);

  // setup a full canvas clip space quad
  const updatePositionBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, updatePositionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1,
     1, -1,
    -1,  1,
    -1,  1,
     1, -1,
     1,  1,
  ]), gl.STATIC_DRAW);

  // setup an id buffer
  const particleTexWidth = 10;
  const particleTexHeight = 10;
  const numParticles = particleTexWidth * particleTexHeight;
  const ids = new Array(numParticles).fill(0).map((_, i) => i);
  const idBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, idBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(ids), gl.STATIC_DRAW);

  // we're going to base the initial positions on the size
  // of the canvas so lets update the size of the canvas
  // to the initial size we want
  webglUtils.resizeCanvasToDisplaySize(gl.canvas);

  // create random positions and velocities.
  const positions = new Float32Array(
      ids.map(_ => [rand(canvas.width), rand(canvas.height), 0, 0]).flat());
  const velocities = new Float32Array(
      ids.map(_ => [rand(canvas.width), rand(canvas.height), 0, 0]).flat());
      //ids.map(_ => [rand(-300, 300), rand(-300, 300), 0, 0]).flat());
  // create a texture for the velocity and 2 textures for the positions.
  let velocityTex = createTexture(gl, velocities, particleTexWidth, particleTexHeight);
  let velocityTex2 = createTexture(gl, null, particleTexWidth, particleTexHeight);
  let positionTex1 = createTexture(gl, positions, particleTexWidth, particleTexHeight);
  let positionTex2 = createTexture(gl, null, particleTexWidth, particleTexHeight);

  // create 2 framebuffers. One that renders to positionTex1 one
  // and another that renders to positionTex2

  let velocityFB1 = createFramebuffer(gl, velocityTex);
  let velocityFB2 = createFramebuffer(gl, velocityTex2);

  let positionsFB1 = createFramebuffer(gl, positionTex1);
  let positionsFB2 = createFramebuffer(gl, positionTex2);

  /*
  let oldPositionsInfo = {
    fb: positionsFB1,
    tex: positionTex1,
  };
  let newPositionsInfo = {
    fb: positionsFB2,
    tex: positionTex2,
  };
  */

  let then = 0;
  function render(time) {
    // convert to seconds
    time *= 0.001;
    // Subtract the previous time from the current time
    const deltaTime = time - then;
    // Remember the current time for the next frame.
    then = time;

    webglUtils.resizeCanvasToDisplaySize(gl.canvas);


    // render to the new velocities
    //gl.bindFramebuffer(gl.FRAMEBUFFER, newPositionsInfo.fb);
    gl.bindFramebuffer(gl.FRAMEBUFFER, velocityFB2);
    gl.viewport(0, 0, particleTexWidth, particleTexHeight);

    // setup our attributes to tell WebGL how to pull
    // the data from the buffer above to the position attribute
    // this buffer just contains a -1 to +1 quad for rendering
    // to every pixel
    gl.bindBuffer(gl.ARRAY_BUFFER, updateVelocitiesBuffer);
    gl.enableVertexAttribArray(updateVelocityPrgLocs.velocity);
    gl.vertexAttribPointer(
        updateVelocityPrgLocs.velocity,
        2,         // size (num components)
        gl.FLOAT,  // type of data in buffer
        false,     // normalize
        0,         // stride (0 = auto)
        0,         // offset
    );

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, velocityTex);

    gl.useProgram(updateVelocityProgram);
    gl.uniform1i(updateVelocityProgram.velocityTex, 0);  // tell the shader the position texture is on texture unit 1
    gl.uniform2f(updateVelocityProgram.texDimensions, particleTexWidth, particleTexHeight);
    gl.uniform1f(updateVelocityProgram.deltaTime, deltaTime);

    gl.drawArrays(gl.TRIANGLES, 0, 6);  // draw 2 triangles (6 vertices)



    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    // setup our attributes to tell WebGL how to pull
    // the data from the buffer above to the id attribute
    gl.bindBuffer(gl.ARRAY_BUFFER, idBuffer);
    gl.enableVertexAttribArray(drawParticlesProgLocs.id);
    gl.vertexAttribPointer(
        drawParticlesProgLocs.id,
        1,         // size (num components)
        gl.FLOAT,  // type of data in buffer
        false,     // normalize
        0,         // stride (0 = auto)
        0,         // offset
    );

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, velocityTex2);

    gl.useProgram(drawParticlesProgram);
    gl.uniform2f(drawParticlesProgLocs.texDimensions, particleTexWidth, particleTexWidth);
    gl.uniform1i(drawParticlesProgLocs.positionTex, 0);  // tell the shader the position texture is on texture unit 0
    gl.uniformMatrix4fv(
        drawParticlesProgLocs.matrix,
        false,
        m4.orthographic(0, gl.canvas.width, 0, gl.canvas.height, -1, 1));

    gl.drawArrays(gl.POINTS, 0, numParticles);



    // render to the new positions
    //gl.bindFramebuffer(gl.FRAMEBUFFER, newPositionsInfo.fb);
    gl.bindFramebuffer(gl.FRAMEBUFFER, positionsFB2);
    gl.viewport(0, 0, particleTexWidth, particleTexHeight);

    // setup our attributes to tell WebGL how to pull
    // the data from the buffer above to the position attribute
    // this buffer just contains a -1 to +1 quad for rendering
    // to every pixel
    gl.bindBuffer(gl.ARRAY_BUFFER, updatePositionBuffer);
    gl.enableVertexAttribArray(updatePositionPrgLocs.position);
    gl.vertexAttribPointer(
        updatePositionPrgLocs.position,
        2,         // size (num components)
        gl.FLOAT,  // type of data in buffer
        false,     // normalize
        0,         // stride (0 = auto)
        0,         // offset
    );

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, positionTex1);
    gl.activeTexture(gl.TEXTURE0 + 1);
    gl.bindTexture(gl.TEXTURE_2D, velocityTex2);

    gl.useProgram(updatePositionProgram);
    gl.uniform1i(updatePositionPrgLocs.positionTex, 0);  // tell the shader the position texture is on texture unit 0
    gl.uniform1i(updatePositionPrgLocs.velocityTex, 1);  // tell the shader the position texture is on texture unit 1
    gl.uniform2f(updatePositionPrgLocs.texDimensions, particleTexWidth, particleTexHeight);
    gl.uniform2f(updatePositionPrgLocs.canvasDimensions, gl.canvas.width, gl.canvas.height);
    gl.uniform1f(updatePositionPrgLocs.deltaTime, deltaTime);

    gl.drawArrays(gl.TRIANGLES, 0, 6);  // draw 2 triangles (6 vertices)


/*
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    // setup our attributes to tell WebGL how to pull
    // the data from the buffer above to the id attribute
    gl.bindBuffer(gl.ARRAY_BUFFER, idBuffer);
    gl.enableVertexAttribArray(drawParticlesProgLocs.id);
    gl.vertexAttribPointer(
        drawParticlesProgLocs.id,
        1,         // size (num components)
        gl.FLOAT,  // type of data in buffer
        false,     // normalize
        0,         // stride (0 = auto)
        0,         // offset
    );

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, positionTex2);

    gl.useProgram(drawParticlesProgram);
    gl.uniform2f(drawParticlesProgLocs.texDimensions, particleTexWidth, particleTexWidth);
    gl.uniform1i(drawParticlesProgLocs.positionTex, 0);  // tell the shader the position texture is on texture unit 0
    gl.uniformMatrix4fv(
        drawParticlesProgLocs.matrix,
        false,
        m4.orthographic(0, gl.canvas.width, 0, gl.canvas.height, -1, 1));

    gl.drawArrays(gl.POINTS, 0, numParticles);
*/


    // swap which texture we will read from
    // and which one we will write to

    /*const tempv1 = velocityFB1;
    velocityFB1 = velocityFB2;
    velocityFB2 = tempv1;

    const tempv2 = velocityTex;
    velocityTex = velocityTex2;
    velocityTex2 = tempv2;*/

    const temp1 = positionsFB1;
    positionsFB1 = positionsFB2;
    positionsFB2 = temp1;

    const temp2 = positionTex1;
    positionTex1 = positionTex2;
    positionTex2 = temp2;
    //const temp = oldPositionsInfo;
    //oldPositionsInfo = newPositionsInfo;
    //newPositionsInfo = temp;


    if (run) {
      requestAnimationFrame(render);
    }
  }
  requestAnimationFrame(render);
}

let run = true;
main();

function stop() {
  run = false;
}

function restart() {
  run = false;
  run = true;
  main();
}
