var {dense, rnn, conv} = require('./topo.js')
var $ = require('./utils.js')
var fs = require('fs')
var tab = require('typedarray-to-buffer')
const tf = $.tf

var mnist = require('./data.js') 

var batch_size =64//256 * 4
var epochas = 12*2
var sample_count = 10000 // using 10k training samples
var input_shape = [batch_size,784]
var z = 10
/* TODO
z_mean and z_dev shall become small, depthy rnns
convoluted idea:  use of conways GoL to colonize image around densities 
*/

var convo = conv({input_shape, layers:[{size: [1,1], depth: 9}, {size: [3, 3], depth:3, activation: 'relu'}, {size: [9, 9], depth: 1, activation: 'relu'}]})

var z_mean = dense({input_shape, layers: [{size: 10},{size: z, activation: 'erf', init: 'orthoNormal', trainable:true}]})
var z_dev = dense({input_shape, layers: [{size: 10},{size: z, activation: 'erf', init: 'orthoNormal', trainable:true}]})

var decode_layers = [{size: input_shape[1], activation: 'relu'}]
var decoder = dense({input_shape: [batch_size, z], layers: decode_layers})

var rconvo = conv({input_shape: decoder.outputShape, layers:[{size: [9,9], depth: 9, activation: 'relu'},{size: [3, 3], depth:3, activation: 'relu'}, {size: [1, 1], depth: 1, activation: 'relu'}]})


var rate = .01
var optimizer = tf.train.adam(rate)
// run it
load_and_run()

async function load_and_run(){
  await mnist.loadData()
  train()
  test()
}

function feed_fwd(input, train, size){
  var conv = convo.flow(input, train) 
  conv = conv.reshape([size || batch_size, input_shape[1]])
  
  let m = z_mean.flow(conv, train)
  let d = z_dev.flow(conv, train) 

  // sample from mean and deviation
  let sample = $.initializers.randomNormal({shape: m.shape, trainable: false}).mul(d).add(m)
  
  var result = rconvo.flow(decoder.flow(sample, train).reshape([size || batch_size, 28, 28, 1]), train)
  
  return {result,  m, d}
}

function train(batch){
  var batch = []
  var labels = []
  for(var x = 0; x < sample_count / batch_size; x++){
    var d = mnist.nextTrainBatch(batch_size)
    batch.push(d.image)
    labels.push(d.label)
  }
  for(var x = 0; x < epochas; x++){
      tf.tidy(() => {
        var _loss = $.scalar(0)
        var dispose = []
        batch.forEach((input, i) => {
          _loss = _loss.add(optimizer.minimize(function(){
            let {result, encoding, m, d} = feed_fwd(input, true)
            var reconLoss = tf.losses.meanSquaredError(input, result)
            var KL_loss = tf.sum($.scalar(1).add(d).sub(tf.square(m)).sub(tf.exp(d)),1).mul($.scalar(-.5))
            var totes = tf.mean(KL_loss.add(reconLoss))//.add(regen)
            //let reg = [convo, rconvo, decoder].reduce((reg, topo) => reg.add(topo.regularize()), $.scalar(0))
            if(i % 10 == 0){ 
              //console.log(`current regularario  is: ${regen.dataSync()}`)
              //console.log(`current loss is: ${totes.dataSync()}`)
            } 
            $.dispose([totes, m, d, KL_loss])
            return totes//.add(reg)
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
  var correct = 0
  var wrong = 0
  for(var x = 0; x < 10; x++){
    var d = mnist.nextTrainBatch(1)
    batch.push(d.image)
    labels.push(d.label)
  }

  batch.forEach((input, i) => {
    var {result} = feed_fwd(input, false, 1)
    /*
    let loss = tf.losses.softmaxCrossEntropy(labels[i], encoding)
    console.log('*****************************************************************')
    let p = tf.argMax(encoding, 1).dataSync()[0]
    let a = tf.argMax(labels[i], 1).dataSync()[0]
    if(p==a) correct++
    else wrong++
    let m = `precidicted: ${p} \nactual: ${a}`
    console.log(m)
    encoding.print()
    labels[i].print()
    tf.mean(loss).print()
    */
    var name = `pic-${i}.raw`
//    draw(input, 'input-' + name)
    draw(result, 'result-' + name)
  })
  console.log(`correct: ${correct}, wrong: ${wrong}, percentage: ${(correct/(correct+wrong))}`)
  
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


