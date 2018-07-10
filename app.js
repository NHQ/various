var dense = require('./dense.js')
var $ = require('./utils.js')
var fs = require('fs')
var tab = require('typedarray-to-buffer')
const tf = $.tf

var mnist = require('./data.js') 

var batch_size = 1000 //256 * 4
var epochas = 120 
var sample_count = 10000 // using 10k training samples
var input_shape = [batch_size,784]

// "dense" returns a sequential multi-layer dense network; we create 2, one for encoder, one for decoder
// tryint to reproduce https://stats.stackexchange.com/questions/190148/building-an-autoencoder-in-tensorflow-to-surpass-pca#answer-307746

var encode_layers = [{size: 1024 * 2, activation: 'linear', init: 'orthoUniform', trianable: false},{size: 1024, activation: 'tanh'},{size: 512, activation: 'sigmoid'}, {size: 128, activation: 'sigmoid'}, {size: 10, activation: 'linear'}]

var z_layer = $.initializers.orthoUniform({shape: [10,10], min: -1, max: 1})

var decode_layers = [{size: 128, activation: 'sigmoid'}, {size: 512, activation: 'sigmoid'},{size: 1024, activation: 'tanh'},{size: 1024 * 2, activation: 'tanh'}, {size: 784, activation: 'linear'}]

var encoder = dense({input_shape, layers: encode_layers, ortho: true})
var decoder = dense({input_shape: encoder.outputShape, layers: decode_layers})

var rate = 1.11
var optimizer = tf.train.adam(rate)
// run it
load_and_run()

async function load_and_run(){
  await mnist.loadData()
  train()
  test()
}

function feed_fwd(input){
  var encoding = encoder.flow(input)
  encoding = encoding.matMul(z_layer)
  var result = decoder.flow(encoding)
  return {result, encoding}
}

function train(batch){
  var batch = []
  var labels = []
  for(var x = 0; x < sample_count / batch_size; x++){
    var d = mnist.nextTrainBatch(batch_size)
    batch.push(d.image.reshape([batch_size, input_shape[1]]))
    labels.push(d.label.reshape([batch_size, 10]))
  }

  for(var x = 0; x < epochas; x++){
    var _loss
    batch.forEach((input, i) => {
      _loss = optimizer.minimize(function(){
        let {result, encoding} = feed_fwd(input)
        let reconLoss = tf.losses.meanSquaredError(input, result)
        let encodeLoss = tf.losses.meanSquaredError(labels[i], encoding)
        if(i % 10 == 0){ // print loss evey 500 train
          console.log('***************************************************************')
          console.log(`current encode loss is: ${encodeLoss.dataSync()[0]}`)
          console.log(`current reconstruction loss is: ${reconLoss.dataSync()[0]}`)
          console.log(`tf memory is ${JSON.stringify(tf.memory())}`)
        } 
        return tf.square(reconLoss).add(encodeLoss)
      }, true)
    })
    console.log(`loss after epoch ${(x+1)}: ${_loss.dataSync()[0]}`)
    mnist.resetTraining()
  }
}

function test(input){
 // TODO: update for node backend 
  // reconstruct a few digits

  var batch = []
  var labels = []
  mnist.resetTraining()
  for(var x = 0; x < 11; x++){
    var d = mnist.nextTestBatch(1)
    batch.push(d.image.reshape([1, input_shape[1]]))
    labels.push(d.label.reshape([1, 10]))
  }
  
  batch.forEach((input, i) => {
    var {result, encoding} = feed_fwd(input)
    let loss = tf.losses.meanSquaredError(labels[i], encoding)
    loss.print()
    var name = `pic-${i}.raw`
    draw(input, 'input-' + name)
    draw(result, 'result-' + name)
  })
}

// render a tensor to canvas and append
function draw(input, name){
  //var canvas = document.createElement('canvas')
  //canvas.width = canvas.height = Math.sqrt(input_shape[1])
  //var ctx = canvas.getContext('2d')
  var data = input.dataSync()
  var imgData = new Int8Array(data.length * 4) //ctx.createImageData(canvas.width, canvas.height)
  for(var x = 0; x < data.length; x++){
    let i = x * 4
    imgData[i] = Math.floor(data[x] * 255)
    imgData[i+1] = Math.floor(data[x] * 255)
    imgData[i+2] = Math.floor(data[x] * 255)
    imgData[i+3] = 255//Math.floor(data[x] * 255)
  }
  fs.writeFileSync('./public/' + name, tab(imgData))
}


