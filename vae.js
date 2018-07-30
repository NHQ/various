var {dense, rnn, conv} = require('./topo.js')
var $ = require('./utils.js')
const tf = $.tf

var mnist = require('./data.js') 

//  training variables
var batch_size = 56
var epochas = 6
var sample_count = 10000 // using 10k training samples
var input_shape = [batch_size,784]


// 3 layer basic convolution (size is kernal, depth is number of filters)
var convo = conv({input_shape, layers:[{size: [9, 9], depth:64}, {size: [1,1], depth: 1}]})

// layers for the dense network
var encode_layers = [{size: 64, depth:9}, {size: 10, activation: 'linear'}]

// dense encoder
var encoder = rnn({input_shape, layers: encode_layers})

var decoder = dense({input_shape: encoder.outputShape, layers: [{size: input_shape[1], activation:'linear'}]})

var deconv = conv({input_shape, layers: [{size: [9,9], depth: 64, transpose: true}, {size: [1,1], depth: 1, transpose: true}]})

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

  // flatten for dense layers
  conv = conv.reshape([size || batch_size, input_shape[1]])

  var encoding = encoder.flow(conv, train) 
  
  var result = deconv.flow(decoder.flow(encoding, train).reshape(input.shape), train)

  return {result, encoding}
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
      var _loss
      var dispose = []
      batch.forEach((input, i) => {
        _loss = optimizer.minimize(function(){
          let {result, encoding} = feed_fwd(input, true)
          let reg = encoder.regularize().add(convo.regularize())
          let reconLoss = tf.abs(tf.mean(tf.sum(tf.sub(input, result), 1)))
          //let reconLoss = tf.sum(tf.add(tf.mul(input, tf.log(tf.add($.scalar(1e-8), result))), tf.mul(tf.sub($.scalar(1), input), tf.log(tf.add($.scalar(1e-8), tf.sub($.scalar(1), result))))), 1)
          let encodeLoss = tf.mean(tf.losses.softmaxCrossEntropy(labels[i], encoding))
          if(i % 10 == 0){ // print loss every ith train
            console.log('***************************************************************')
            console.log(`current encode loss is: ${tf.mean(encodeLoss).dataSync()}`)
            console.log(`current recon loss is: ${reconLoss.dataSync()}`)
           // encoding.print()
           
          } 
          // garbage collection 
          $.dispose([encodeLoss, encoding])

          return encodeLoss.add(reconLoss)
        }, true)
      })

      console.log(`\ntf memory is ${JSON.stringify(tf.memory())}`)
      console.log(`loss after epoch ${(x+1)}: ${_loss.dataSync()[0]}\n`)
      
      mnist.resetTraining()
      $.dispose(dispose, true)
    })
  }
}

function test(input){

  var batch = []
  var labels = []
  mnist.resetTest()
  var correct = 0
  var wrong = 0
  for(var x = 0; x < 1000; x++){
    var d = mnist.nextTrainBatch(1)
    batch.push(d.image)
    labels.push(d.label) 
  }
  
  batch.forEach((input, i) => {
    var {encoding} = feed_fwd(input, false, 1)
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
  })
  console.log(`correct: ${correct}, wrong: ${wrong}, percentage: ${(correct/(correct+wrong))}`)
}

