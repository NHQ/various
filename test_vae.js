var dense = require('./dense.js')
var $ = require('./utils.js')
const tf = $.tf

var input_shape = [1,784]
var mag = 1024
let latent_spacetime = [16, 16]

var lens = tf.variable($.orthoUniform({shape: [input_shape[1], mag], min:-1, max: 1}))
var abstract_mean = tf.variable($.orthoUniform({shape: latent_spacetime}))
var abstract_deviation =tf.variable($.orthoUniform({shape: latent_spacetime}))

console.log('abstract mean***********************************************')
abstract_mean.print()
console.log('abstract devitation***********************************************')
abstract_deviation.print()
var decode_topology = [512, 128, 16]
var encode_topology = Array.from(decode_topology).reverse()
encode_topology.push(input_shape[1])

var variables = []
var decoder = dense({input_shape: [lens[0], mag], topology: decode_topology, ortho: false})
var encoder = dense({input_shape: [input_shape[0], latent_spacetime[1]], topology: encode_topology, ortho: false})

variables = variables.concat(decoder.vars)
variables = variables.concat(encoder.vars)
var rate = .1
var optimizer = tf.train.adamax(rate)

import {MnistData} from './mnist_data'
var data
async function load(){
  data = new MnistData()
  await data.load()
  var batch = []
  for(var x = 0; x < 1000; x++){
    batch.push(data.nextTrainBatch(1))
  }
  train(batch)
  //batch.forEach(e => train(e.xs))
}

//train(tf.ones([1,784]))

load()
/*
tf.tidy(() => {
  train(input)
})
*/

variables.concat(lens, abstract_mean, abstract_deviation)
function train(batch){
  var f = input => {
    var output = decoder.topo(input.dot(lens))
    var mean = output.dot(abstract_mean)
    var deviation = output.dot(abstract_deviation)
    // sample from random normal 
    var norm = $.randomNormal({shape: [mean.shape[0], latent_spacetime[1]]})
    var sample = mean.add(tf.exp(tf.scalar(.5).mul(deviation)).mul(norm))
    var result = encoder.topo(sample)
    /*
    console.log('sample***********************************************')
    sample.print()
    console.log('output***********************************************')
    output.print()
    console.log('result***********************************************')
    result.print()
    console.log('mean***********************************************')
    mean.print()
    console.log('deviation***********************************************')
    deviation.print()
    */
    return {result, mean, deviation} 
  }
  var t = 1
  var totes = 1
  var loss = function(input, {result, mean, deviation}){
    t++
    //console.log(`**********************************${(t)}****************************'`)
    console.log('***********recon LOSS***********************************************')
    var reconstruction_loss = tf.losses.meanSquaredError(input, result) //tf.sqrt(tf.mean(tf.square(input.sub(result))))
    reconstruction_loss.print()
    console.log('***********KL LOSS***********************************************')
    var kl_loss = tf.scalar(-.5).mul(tf.sum(tf.scalar(1).add(deviation).sub(tf.square(mean)).sub(tf.square(tf.exp(deviation)))))
    kl_loss.print()
    var total_loss = reconstruction_loss.add(kl_loss)
    totes = total_loss.dataSync()[0]
    console.log('***********TOTEs***********************************************')
    console.log(totes)
    return total_loss
  } 

  while(t < 5000)
  batch.forEach(input => { 
    optimizer.minimize(() => loss(input.xs, f(input.xs)), false, variables)
   })
}



