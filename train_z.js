var {dense, rnn} = require('./topo.js')
var $ = require('./utils.js')
var fs = require('fs')
var tab = require('typedarray-to-buffer')
const tf = $.tf

var mnist = require('./data.js') 

var batch_size = 55//256 * 4
var epochas = 1024
var sample_count = 10000 // using 10k training samples
var input_shape = [batch_size,784]

// "dense" returns a sequential multi-layer dense network; we create 2, one for encoder, one for decoder
// tryint to reproduce https://stats.stackexchange.com/questions/190148/building-an-autoencoder-in-tensorflow-to-surpass-pca#answer-307746
var lens = {size: 1024 * 2, activation: 'linear', init: 'orthoUniform', trianable: false}
var encode_layers = [{size: 1024, activation: 'sigmoid'},{size: 512, activation: 'sigmoid'}, {size: 256, activation: 'sigmoid'},{size: 128, activation: 'sigmoid'}, {size: 10, activation: 'linear'}]

var z_mean = $.initializers.orthoUniform({shape: [10,100], min: 0, max: 1})
var z_dev= $.initializers.orthoUniform({shape: [10,100], min: 0, max: 1})

var l = 8
var d = new Array(l).fill(0)
decode_layers = d.map((e, i) => ({size: Math.min(input_shape[1], Math.floor(input_shape[1] / (l-1)) * (i +1))}))
decode_layers[l-1].activation = 'linear'

//var decode_layers = [{size: 128, activation: 'linear'}, {size: 512, activation: 'sigmoid'},{size: 1024, activation: 'tanh'},{size: 1024 * 2, activation: 'tanh'}, {size: 784, activation: 'linear'}]

//var lensing = rnn({input_shape, layers: [lens]})
var encoder = rnn({input_shape, depth:4, layers: encode_layers, ortho: true, xav:true})
var decoder = dense({input_shape: encoder.outputShape, layers: decode_layers, xav:true})
var rate = .01
var optimizer = tf.train.adam(rate)
// run it
load_and_run()

function load_and_run(){
  train()
  test()
}

function feed_fwd(input, train){
  var encoding = encoder.flow(input, train)
  //console.log(encoder)
  var z = null// encoding.matMul(z_layer)
  //var result = decoder.flow(encoding)
  return {encoding, z}
}

function train(batch){
  var batch = []
  var labels = []
  for(var x = 0; x < sample_count / batch_size; x++){
    var d = $.variable({init: 'randomUniform', shape: [batch_size, input_shape[1]], trainable: false}).layer
    batch.push(d)
  }

  for(var x = 0; x < epochas; x++){
      tf.tidy(() => {
        var _loss = $.scalar(0)
        var dispose = []
        batch.forEach((input, i) => {
          _loss = _loss.add(optimizer.minimize(function(){
            let {encoding} = feed_fwd(input, true)
            let m = encoding.dot(z_mean)
            let d = encoding.dot(z_dev)
            //let ren = encoder.regularize()
            let regen = encoder.variables.reduce((a, e) => tf.add($.regularize({input: e, l:.001, ll:.001}), a), $.scalar(0)).mul($.scalar(1/input_shape[0]))

            //let red = decoder.regularize()
            var loss = tf.mean($.scalar(.5).add(tf.sum($.scalar(1).add(d).sub(tf.square(m)).sub(tf.square(tf.exp(d))), -1)))
            var totes = tf.abs(loss).add(regen)
            if(i % 10 == 0){ // print loss evey 500 train
              console.log(`current regularario  is: ${regen.dataSync()}`)
              console.log(`current loss is: ${totes.dataSync()}`)
            } 
            $.dispose([totes, m, d, loss, regen])
            return totes
          }, true))
        })
      //_loss.print()
      console.log('********************************************')
      console.log(`tf memory is ${JSON.stringify(tf.memory())}`)
      console.log(`average loss after epoch ${(x+1)}:`) 
      _loss.div($.scalar(batch.length)).print()
      mnist.resetTraining()
      $.dispose(dispose, true)
      dispose = []//encoder.disposal.map(e => false).filter(Boolean)
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


