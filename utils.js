var fs = require('fs')
Math.random = require('math-random')
Error.stackTraceLimit = Infinity
const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node-gpu')
//require('@tensorflow/tfjs-node-gpu')
var atob = require('arraybuffer-to-buffer')
var $ = require('./cheatcode.js')

tf.linear = rootOp
var log = console.log

const init = initializers = {harmonic, orthoNormal, orthoUniform, orthoTruncated, randomNormal, randomUniform, randomTruncated, zeros, ones}

module.exports = {tautime, log, jsdft, dft, harmonic, phase, mag, tf, conv2d, gc, regularize, scalar, dispose, variable, initializers, init, combinatorial, nextTick, createRollMatrix, assert, a0, invertMask}

function rootOp(input){return input}

const scalars = {}
var disposal = []

function phase(a,b){
  return tf.atan2(a, b)//.div(scalar(Math.PI).div(scalar(2)))
}

function mag(a,b){
  return tf.sqrt(tf.square(a).add(tf.square(b)))
}

function a0(x){
  return Array(x).fill(0)
}

function conv2d(params){
  params = configur8(params)
  let {layer, activation, saver} = variable(params)
  let pool = function(input){ return params.pool.fn(input, params.pool.size, params.pool.strides, params.pool.pad)}
  let conv = params.transpose && false ? tf.conv2dTranspose : tf.conv2d
  return {filter: function(input){
    return activation(pool(conv(input, layer, params.strides, params.pad, 'NHWC', params.dilations)))
  }, layer, pool, activation, saver}
}


function gc(g=[], run=false){
  dispose(g, run)
  return g
}

function dispose(gc=[], run=false){
  if(gc.length) disposal = disposal.concat(gc)
  if(run) {
    tf.dispose(disposal)
    for(x in disposal) disposal.shift()
  }
}


function scalar(s, type='float32', train=false){
  if(scalars[s]) return scalars[s]
  else { 
    scalars[s] = tf.variable(tf.scalar(s, type), train) 
    return scalars[s]
  }
}

function assert(thing, whiches){
  for(which in whiches) {
    if(thing[whiches[which]] == undefined){
      throw new Error(`¡this thing ${thing.toString()} has not ${whiches[which]}!`)
    }
  }
}

function configur8({
  trainable=true, init='randomNormal', min=0, max=1, mean=0, dev=1, regularizer=false, activation='relu', type='float32', 
  strides=[1,1], pad='same', dilations=[1,1], 
  defaultPool={fn: rootOp, size: 1, strides: 1, pad: 'same'}
}){
  // the idea here is to add these defined params to a configration that lacks them
  // so that from this point on, the API has uniform arrity and zero missing params...
  var config = arguments['0']
  config['trainable'] = trainable
  config['init'] = init
  config['mean'] = mean 
  config['dev'] = dev
  config['min'] = min 
  config['max'] = max
  config['type'] = type
  config['regularizer'] = regularizer
  config['activation'] = activation
  config['pad'] = pad 
  config['strides'] = strides 
  config['dilations'] = dilations
  //config['pool'] = (config.pool && true) ? {...defaultPool, ...config.pool} : defaultPool
  //assert(config, 'shape')
  //assert(config, 'layers')
  return config
}

function regularize({input, l1=true, l2=true, l=.01, ll=.01}){
  assert(arguments['0'], ['input'])
  let r = scalar(0)
  if(l1) r = tf.add(r, tf.sum(tf.mul(scalar(l), tf.abs(input))))
  if(l2) r = tf.add(r, tf.sum(tf.mul(scalar(ll), tf.square(input))))
  return r
}

function combinatorial (n, k){ var p = n - k; var x = 1; while(n > p) x*=(n--); return x}

function variable(config){
  let params = configur8(config)
  let init = initializers[params.init]
  let activation = tf[params.activation] 
  var layer
  var pid = undefined
  if(params.id){ // try load
    try{
      fs.accessSync(pid = './filters/' + params.id)
      console.log(params)
      var buf = fs.readFileSync(pid)
      console.log(buf.length, buf.buffer.byteLength)
      buf = new Float32Array(buf.buffer.slice(0,buf.length))
      layer = tf.tensor(buf, params.shape, params.type)
      console.log('loaded layer id: ' + params.id)
    } catch (err){
      console.log(err)
      layer = init(params)
    }
  } else layer = init(params)
  layer = tf.variable(layer, params.trainable)
  var saver = function(){
    //console.log(params.id, layer)
    fs.writeFile('./filters/' + params.id, atob(layer.dataSync().buffer), function(e,r){
      if(e) console.log(e)
    })
  }
  //if(params.id) save()
  return {layer,  activation, saver}
}

async function nextTick(fn){ await tf.nextFrame(); fn()}

function zeros({shape, type}){
  return tf.zeros(shape, type)
}

function ones({shape, type}){
  return tf.ones(shape, type)
}

function randomNormal({shape, mean=0, dev=1, type='float32'}){
  return tf.randomNormal(shape, mean, dev, type)
}

function randomTruncated({shape, mean=0, dev=1, type='float32'}){
  return tf.randomNormal(shape, mean, dev, type)
}

function randomUniform({shape, min=-1, max=1, type='float32'}){
  return tf.randomUniform(shape, min, max, type)
}

function orthoNormal({shape, mean=0, dev=1, type='float32'}){
  return tf.linalg.gramSchmidt(tf.randomNormal(shape, mean, dev, type))
}

function orthoTruncated({shape, mean=0, dev=1, type='float32'}){
  return tf.linalg.gramSchmidt(tf.truncatedNormal(shape, mean, dev, type))
}

function orthoUniform({shape, min=-1, max=1, type='float32'}){
  return tf.linalg.gramSchmidt(tf.randomUniform(shape, min, max, type))
}

function harmonic({base=27.5, size=100, shape=[1,100]}){
  let y = new Float32Array(size)
  for(var x = 0; x < size; x++){
    y[x] = base * Math.pow(2, x/12)
  }
  return tf.tensor(y, shape)
}

function tautime(z, sr){
  var t = tf.tensor(Array(z).fill(0).map((e,i)=> i / sr), [1, z]).reshape([z,1])//, z, z).div($.scalar(sampleRate))
  return t.mul(scalar(Math.PI * 2))
}

function dft(t, f){
  let y = tf.neg(t.matMul(f))
  let s = tf.sin(y)
  let c = tf.cos(y)
  let sin = $ => $.matMul(s)
  let cos = $ => $.matMul(c)
  return {cos, sin}

}

function dft(t, f){
  var y = tf.neg(t.matMul(f))
  let sin = $ => $.matMul(tf.sin(y))
  let cos = $ => $.matMul(tf.cos(y))
  return {cos, sin}

}
function jsdft(x, k, sr){
   
  let y = x.map((e,i)=>[e*Math.cos(-(Math.PI * 2 * i * k / sr)), e*Math.sin(-(Math.PI * 2 * k * i /sr))])
  return y
}

// inverts a one-hot mask (such as the identity matrix, or eye)
function invertMask(mask){
  return mask.add(scalar(1)).sub(scalar(2)).mul(scalar(-1))
}

function createRollMatrix(s, t){

  return roll(s, t)

  function roll(s, t){ 
    l = s * s
    var one = t > 0 ? rollRightOne(l) : rollLeftOne(Math.abs(l))
    var rm = tf.eye(Math.sqrt(l))
    for(var x = 0; x < Math.abs(t); x++){
       rm = tf.matMul(rm, one)
    }
    return rm
  }

  function rollLeftOne(l){
    var a = new Float32Array(l)
    var n = Math.sqrt(l)
    a.fill(0)
    a.forEach((e,i,a) => (i - n) % (n + 1) === 0 ? a[i] = 1 : a[i] = 0)
    a[n - 1] = 1
    return tf.tensor(a, [n,n], 'float32')
  }

  function rollRightOne(l){
    var a = new Float32Array(l)
    var n = Math.sqrt(l)
    a.fill(0)
    a.forEach((e,i,a) => (i - 1) % (n + 1) === 0 ? a[i] = 1 : a[i] = 0)
    a[l - n] = 1
    return tf.tensor(a, [n,n], 'float32')
  }

}

