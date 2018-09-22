var {dense, rnn, conv} = require('./topo.js')
var $ = require('./utils.js')
var load = require('./load.js')
var fs = require('fs')
var tab = require('typedarray-to-buffer')
var atob = tab
const tf = $.tf

var batch_size =1 //256 * 4
var epochas = 1.00000
var binSize = 2048 / 8 
var manifold = 1
var dimension = 1//2
var input_shape = [batch_size, manifold, binSize, dimension]
var z = 32 

var nextBatch = require('./load.js')() // defaults is ./data dir & batchSize = dirs.length

/* TODO
z_mean and z_dev shall become small, depthy rnns
convoluted idea:  use of conways GoL to colonize image around densities 
*/

let cnn_layers = [{size: [3,3], depth:32, activation: 'relu'}, {size: [9,9], depth:16}, {size: [27,27], depth: 8}].reverse()

let cnn = conv({input_shape: [batch_size, 1311, binSize, dimension ], layers: cnn_layers})

let rnn_layers = [{size: z, depth:1, activation: 'sigmoid'}]

var z_mean = dense({input_shape: [batch_size * binSize, z], layers: rnn_layers })
var z_dev = dense({input_shape: [batch_size * binSize, z], layers: rnn_layers})

var decode_layers = [{size: binSize, depth: 1}]
var decoder = rnn({input_shape: [batch_size, z], layers: decode_layers})

var rate = .0002
var optimizer = tf.train.adam(rate)
// run it
load_and_run()

async function load_and_run(){
//  await mnist.loadData()
  train()
  //test()
}

function feed_fwd(input, train, size){
  let c = cnn.flow(input).reshape([-1, z])
  let m = z_mean.flow(c, train)
  let d = z_dev.flow(c, train) 

  // sample from mean and deviation
  let sample = $.initializers.randomNormal({shape: m.shape, trainable: false}).mul(d).add(m)
  
  var result = decoder.flow(sample.reshape([1311, binSize]), train)//.reshape([].concat([size || batch_size], [manifold, binSize, 1]))
  
  return {result,  m, d, sample}
}


function train(){
  const rollers = [2,3,5,8,13,21]
  const rolls = rollers.map((e,i) => {
    return $.createRollMatrix(binSize, e)
  })
  for(var x = 0; x < epochas; x++){
      tf.tidy(() => {
        var batch = Array(batch_size).fill(0).map(e => nextBatch(binSize))
        let signal = tf.stack(batch.map(e => e.signal))
        let time = tf.stack(batch.map(e => e.time), null)
        var rolled = rolls.map((e, i) => {
          let howmany = Math.floor(binSize / rollers[i])
          let rolled = []
          for(var x = 0; x < howmany; x++){
            rolled.push(signal.matMul(e)) //tf.stack([signal.matMul(e), time.matMul(e)], 2))
          }
          return tf.squeeze(tf.stack(rolled))
        })
        rolled = tf.expandDims(tf.expandDims(tf.concat(rolled, [0]), 0),3)
        //console.log(tf.concat(rolled, [0]))
        //console.log(signal)
        //process.exit()
        var _loss = $.scalar(0)
        var dispose = [signal, time]
        _loss = _loss.add(optimizer.minimize(function(){
          var {result, m, d, sample} = feed_fwd(rolled, true)
  //        result = result.mul(time)
          console.log(result)
          result = tf.unstack(result.slice([0,0], [1]))[0]//.map(e => e.squeeze()) // audio channel only
          result.print()
          signal.print()
          result.sub(signal).print()
          var reconLoss = tf.sqrt(tf.mean(tf.square(tf.sub(tf.squeeze(signal), result))))//tf.losses.meanSquaredError(signal, res)
          var KL_loss = tf.sum($.scalar(1).add(d).sub(tf.square(m)).sub(tf.exp(d)),1).mul($.scalar(-.5))
          var totes = tf.mean(KL_loss.add(reconLoss))//.add(regen)
          //let reg = [z_mean, z_dev, decoder].reduce((reg, topo) => reg.add(topo.regularize()), $.scalar(0))
            //console.log(`current regularario  is: ${reg.dataSync()}`)
//      console.log('\n********************************************')
//            console.log(`current reconstruct loss is: ${reconLoss.dataSync()}`)
//            console.log(`current kl loss is: ${tf.mean(KL_loss).dataSync()}`)
          $.dispose([totes, m, d, KL_loss, sample])
          return totes//.add(reg)
        }, true))
      //_loss.print()
      //console.log(`tf memory is ${JSON.stringify(tf.memory())}`)
      //console.log(`average loss after epoch ${(x+1)}:`) 
      //console.log(_loss.div($.scalar(1)).dataSync()[0] + '\n')
      $.dispose(dispose, true)
      dispose = []//encoder.disposal.map(e => false).filter(Boolean)
    })
  }
}

function test(input){
 // TODO: update for node backend 
  // reconstruct a few digits
const rollers = [2,3,5,8,13,21]
const rolls = rollers.map((e,i) => {
  return $.createRollMatrix(binSize, e)
})
  var keke = require('fs').createWriteStream('./kekekeke2.raw')
  for(var x = 0; x < 3000; x++){
    tf.tidy(() => {
        var batch = Array(batch_size).fill(0).map(e => nextBatch(binSize))
        let signal = tf.stack(batch.map(e => e.signal))
        let time = tf.stack(batch.map(e => e.time), null)
        var rolled = rolls.map((e, i) => {
          let howmany = Math.floor(binSize / rollers[i])
          let rolled = []
          for(var x = 0; x < howmany; x++){
            rolled.push(signal.matMul(e))
          }
          return tf.squeeze(tf.stack(rolled))
        })
        rolled = tf.concat(rolled, [0])
      var {result, m, d} = feed_fwd(rolled, true) 
    //  result = result.mul(time)
      //var reconLoss = tf.losses.meanSquaredError(batch, result)
      res = tf.unstack(result.slice([0,0], [1]))//.map(e => e.squeeze()) // audio channel only
      res.forEach((e, i) => {
        if(i % 1 == 0) keke.write(atob(e.dataSync()))
      }) 
      $.dispose(res)
      $.dispose([result, m, d, batch], true) 
      })
  }

  
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


