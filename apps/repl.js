var $ = require('../utils.js')
var ca = require('cellular-automata')
var ndarray = require('ndarray')
var savePixels = require("save-pixels")
var unsq = require('ndarray-unsqueeze')
var ops = require('ndarray-ops')
var cwise = require('cwise')
var concat = require('ndarray-concat-cols')
var shape = [window.innerHeight/2, window.innerHeight/2]
var px = 10
var g = new ca(shape)
g.fillWithDistribution([[1, 13], [0, 87]]);

var state = $.tf.tensor(new Float32Array(g.array.data), [1,shape[0], shape[1],1])
let game = $.gol(shape, state)
let ctx = canv(shape, px)
let ctz = canv(shape, px)
let imd = ctx.getImageData(0,0,shape[0],shape[1])
let imf = ctz.getImageData(0,0,shape[0],shape[1])
setInterval(play, 1000/2)
async function play(){
  draw()
}
async function draw(){
  let gd = game.next()
  let gda = gd.state.dataSync()
/*  let gdb = g.apply('23/3', 1).array.data
  //console.log(gda, gdb)
  for(var i = 0; i < imf.data.length; i+=4){
    imf.data[i] = imf.data[i+1] = imf.data[i+2] = gdb[Math.floor(i/4)] *255 
    imf.data[i+3] = 255
  }
  ctz.clearRect(0,0, shape[0], shape[1])
  ctz.putImageData(imf, 0,0)
*/
  for(var i = 0; i < imd.data.length; i+=4){
    imd.data[i] = imd.data[i+1] = imd.data[i+2] = gda[Math.floor(i/4)] *255 
    imd.data[i+3] = 255
  }
  ctx.clearRect(0,0, shape[0], shape[1])
  ctx.putImageData(imd, 0,0)
}
function pngol(state){
  let n = ndarray(state.dataSync(), shape)
  ops.mulseq(n, 255)
  let gg = unsq(n)
  return savePixels(concat([gg, gg, gg]), "canvas")//.pipe(process.stdout)

}

async function space(){
  return new Promise(res => {
    document.addEventListener('keydown', e => res(e))
  }, {once: true})
}
function canv(shape, px){
  var canvas = document.createElement('canvas')
  canvas.style.width = canvas.style.height = shape[0]  + 'px'
  canvas.height = shape[1]
  canvas.width = shape[0] 
  document.body.appendChild(canvas)
  var ctx = canvas.getContext('2d')
  ctx.strokeStyle = 'gray' //gradient;
  ctx.lineWidth = .5;
  return ctx
}
function grid(draw, w, h, lifeSize){
  draw.clearRect(0,0,w,h);
  draw.fillStyle = rgba(0,0,0,1)
  draw.fillRect(0,0,w,h)
  draw.strokeStyle = '#fff';
  for(var x = 0; x < w; x+=lifeSize){
    draw.moveTo(x, 0)
    draw.lineTo(x, h)
  }

  for(var y = 0; y < h; y+= lifeSize){
    draw.moveTo(0, y)
    draw.lineTo(w, y)
  }
  draw.stroke()

  function rgba(){
    return 'rgba('+Array.prototype.join.call(arguments, ',')+')'
  }
}

