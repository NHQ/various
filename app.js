var dense = require('./dense.js')
var $ = require('./utils.js')
const tf = $.tf

var mnist = require('./data.js') 

var batch_size = 1
var epochas = 6
var sample_count = 20000 // using 20k training samples
var input_shape = [batch_size,784]

// "dense" returns a sequential multi-layer dense network; we create 2, one for encoder, one for decoder
// tryint to reproduce https://stats.stackexchange.com/questions/190148/building-an-autoencoder-in-tensorflow-to-surpass-pca#answer-307746

var encode_layers = [{size: 512, activation: 'elu'}, {size: 128, activation: 'elu'}, {size: 2, activation: 'linear'}]
var decode_layers = [{size: 128, activation: 'elu'}, {size: 512, activation: 'elu'}, {size: 784, activation: 'sigmoid'}]
var encoder = dense({input_shape, layers: encode_layers})
var decoder = dense({input_shape: encoder.outputShape, layers: decode_layers})

var rate = .01
var optimizer = tf.train.adam(rate)
tf.tidy(()=>{})
// run it
load_and_run()

async function load_and_run(){
  await mnist.loadData()
  train()
  //test()
}

function feed_fwd(input){
  var output = encoder.flow(input)
  var result = decoder.flow(output)
  return result
}

function loss(input, result, i){
  var loss = tf.losses.meanSquaredError(input, result)
  if(i % 500 == 0){ // print loss evey 500 train
    console.log(loss.dataSync()[0])
  } 
  return loss
} 

function train(batch){
  var batch = []

  for(var x = 0; x < sample_count / batch_size; x++){
    batch.push(mnist.nextTrainBatch(batch_size).image.reshape([batch_size,784]))
  }

  for(var x = 0; x < epochas; x++){
    var _loss
    batch.forEach((input, i) => {
      _loss = optimizer.minimize(() => loss(input, feed_fwd(input), i), true)
    })
    console.log(`loss after epoch ${x}: ${_loss.dataSync()[0]}`)
    mnist.resetTraining()
  }
}

function test(input){
 // TODO: update for node backend 
  // reconstruct a few digits

  var batch = []

  for(var x = 0; x < 11; x++){
    batch.push(data.nextTrainBatch(batch_size))
  }
  
  batch.forEach(input => {
    var result = feed_fwd(input.xs)
    //draw(input.xs)
    //draw(result)
  })
}

// render a tensor to canvas and append
function draw(input){
  var canvas = document.createElement('canvas')
  canvas.width = canvas.height = Math.sqrt(input_shape[1])
  var ctx = canvas.getContext('2d')
  var imgData = ctx.createImageData(canvas.width, canvas.height)
  var data = input.dataSync()
  for(var x = 0; x < data.length; x++){
    let i = x * 4
    imgData.data[i] = Math.floor(data[x] * 255)
    imgData.data[i+1] = Math.floor(data[x] * 255)
    imgData.data[i+2] = Math.floor(data[x] * 255)
    imgData.data[i+3] = 255//Math.floor(data[x] * 255)
  }
  ctx.putImageData(imgData, 0, 0)
  document.body.appendChild(canvas)
}


