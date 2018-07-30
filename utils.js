Math.random = require('math-random')
Error.stackTraceLimit = Infinity
const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node-gpu')

tf.linear = rootOp

const init = initializers = {configur8, orthoNormal, orthoUniform, orthoTruncated, randomNormal, randomUniform, randomTruncated}

module.exports = {tf, conv2d, gc, regularize, scalar, dispose, variable, initializers, init, combinatorial, nextTick, createRollMatrix, assert}

function rootOp(input){return input}

const scalars = {}
var disposal = []

function conv2d(params){
  params = configur8(params)
  let vars = variable(params)
  let conv = params.transpose && false ? tf.conv2dTranspose : tf.conv2d
  return {filter: function(input){
    return vars.activation(conv(input, vars.layer, params.strides, params.pad))
  }, vars}
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
  trainable=true, init='randomNormal', min=0, max=1, mean=0, dev=1, regularizer=false, activation='tanh', type='float32', 
  strides=[1,1], pad='same', dilations=[0,0]
}){
  // the idea here is to add these defined params to a configration that lacks them
  // so that from this point on, the API has uniform arrity and zero missing params...
  var config = arguments['0']
  config['trainable'] = trainable
  config['init'] = init
  config['mean'] = mean 
  config['dev'] = dev
  config['min'] = min 
  config['mac'] = max
  config['type'] = type
  config['regularizer'] = regularizer
  config['activation'] = activation
  config['pad'] = pad 
  config['strides'] = strides 
  config['dilations'] = dilations
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
  let layer = tf.variable(init(params), params.trainable)
  return {layer,  activation}
}

async function nextTick(fn){ await tf.nextFrame(); fn()}

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

