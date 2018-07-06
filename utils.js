const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node-gpu')

module.exports = {tf, combinatorial, nextTick, orthoUniform, orthoNormal, orthoTruncated, randomNormal, createRollMatrix}

function combinatorial (n, k){ var p = n - k; var x = 1; while(n > p) x*=(n--); return x}

async function nextTick(fn){ await tf.nextFrame(); fn()}

function orthoUniform({shape, min=-1, max=1, type='float32'}){
  return tf.linalg.gramSchmidt(tf.randomUniform(shape, min, max, type))
}

function orthoTruncated({shape, mean=0, dev=1, type='float32'}){
  return tf.linalg.gramSchmidt(tf.truncatedNormal(shape, mean, dev, type))
}

function randomNormal({shape, mean=0, dev=1, type='float32'}){
  return tf.randomNormal(shape, mean, dev, type)
}

function orthoNormal({shape, mean=0, dev=1, type='float32'}){
  return tf.linalg.gramSchmidt(tf.randomNormal(shape, mean, dev, type))
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

