var {dense, rnn} = require('./topo.js')
var $ = require('./utils.js')
var fs = require('fs')
var tab = require('typedarray-to-buffer')
const tf = $.tf

var mnist = require('./data.js') 

var batch_size = 100//256 * 4
var epochas = 12
var sample_count = 10000 // using 10k training samples
var input_shape = [batch_size,784]

// "dense" returns a sequential multi-layer dense network; we create 2, one for encoder, one for decoder
// tryint to reproduce https://stats.stackexchange.com/questions/190148/building-an-autoencoder-in-tensorflow-to-surpass-pca#answer-307746
var lens = {size: 1024 * 2, activation: 'linear', init: 'orthoUniform', trianable: false}
var encode_layers = [{size: 1024, activation: 'sigmoid'},{size: 512, activation: 'sigmoid'}, {size: 256, activation: 'sigmoid'},{size: 128, activation: 'sigmoid'}, {size: 10, activation: 'linear'}]

//var z_layer = $.initializers.orthoUniform({shape: [10,100], min: 0, max: 1})
var l = 8
var d = new Array(l).fill(0)
decode_layers = d.map((e, i) => ({size: Math.min(input_shape[1], Math.floor(input_shape[1] / (l-1)) * (i +1))}))
decode_layers[l-1].activation = 'linear'

//var decode_layers = [{size: 128, activation: 'linear'}, {size: 512, activation: 'sigmoid'},{size: 1024, activation: 'tanh'},{size: 1024 * 2, activation: 'tanh'}, {size: 784, activation: 'linear'}]

//var lensing = rnn({input_shape, layers: [lens]})
var encoder = rnn({input_shape, layers: encode_layers, ortho: true})
var decoder = dense({input_shape: encoder.outputShape, layers: decode_layers})

var rate = .01
var optimizer = tf.train.adam(rate)
// run it
load_and_run()

async function load_and_run(){
  await mnist.loadData()
  train()
  test()
}

function feed_fwd(input, train){
  var encoding = encoder.flow(input, train)
  //console.log(encoder)
  var z = null// encoding.matMul(z_layer)
  var result = decoder.flow(encoding)
  return {result, encoding, z}
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
    tf.tidy(() => {
      var _loss
      var dispose = []
      batch.forEach((input, i) => {
        _loss = optimizer.minimize(function(){
          let {result, encoding} = feed_fwd(input, true)
          let reconLoss = tf.sum(tf.losses.meanSquaredError(input, result))
          let encodeLoss = tf.sum(tf.losses.softmaxCrossEntropy(labels[i], encoding))
          if(i % 10 == 0){ // print loss evey 500 train
            console.log('***************************************************************')
            console.log(`current encode loss is: ${encodeLoss.dataSync()}`)
            console.log(`current reconstruction loss is: ${reconLoss.dataSync()}`)
          } 
          return reconLoss.add(encodeLoss)
        }, true)
      })
      //_loss.print()
      console.log(`tf memory is ${JSON.stringify(tf.memory())}`)
     // console.log(`loss after epoch ${(x+1)}: ${_loss.dataSync()[0]}`)
      mnist.resetTraining()
      let ld = encoder.disposal.length
      console.log(ld)
      tf.dispose(encoder.disposal)
      for(x in encoder.dispsosal) encoder.disposal.shift()//encoder.disposal.map(e => false).filter(Boolean)
    })
  }
}

function test(input){
 // TODO: update for node backend 
  // reconstruct a few digits

  var batch = []
  var labels = []
  mnist.resetTest()
  for(var x = 0; x < 21; x++){
    var d = mnist.nextTrainBatch(1)
    batch.push(d.image.reshape([1, input_shape[1]]))
    labels.push(d.label.reshape([1, 10]))
  }
  
  batch.forEach((input, i) => {
    var {result, encoding} = feed_fwd(input)
    let loss = tf.losses.softmaxCrossEntropy(labels[i], encoding)
    encoding.print()
    labels[i].print()
    tf.sum(loss).print()
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


