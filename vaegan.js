var {dense, rnn, conv} = require('./topo.js')
var $ = require('./utils.js')
var load = require('./load.js')
var fs = require('fs')
var tab = require('typedarray-to-buffer')
var atob = tab
const tf = $.tf

var epochas =1e3//00000
var binSize = 2048 
var manifold = 1
var dimension = 1//2
var z = 100/10
var fqbins = 100
var sampleRate = 48000


var {nextBatch, batchSize} = require('./load.js')() // defaults is ./data dir & batchSize = dirs.length
var input_shape = [batchSize, binSize]

var time = $.tautime(binSize, sampleRate)
var ttime = time.transpose()
var filter = $.variable({init: 'harmonic', base: 27.5, activation: 'relu', size: fqbins, shape: [1, fqbins], id: 'dftw'})
var dft = $.variable({init: 'harmonic', base: 27.5, activation: 'relu', trainable: false, size: fqbins, id:'dft', shape: [1, fqbins]})

var tx = $.dft(time, filter.layer, sampleRate)

function transform(signal){
  var sin = tx.sin(signal)
  var cos = tx.cos(signal)
  var mag = $.mag(cos, sin).mul($.scalar(2)).div($.scalar(binSize))
  var phase = $.phase(cos, sin).div($.scalar(Math.PI)).div($.scalar(2))
  return {mag, phase}
}

var z_mean = rnn({input_shape: [batchSize, fqbins], layers: [{size: fqbins * 4, init: 'orthoUniform', id:'z_mean', depth: 1}, {size:z, depth:4, activation: 'elu'}]})
var z_dev = rnn({input_shape: [batchSize, fqbins], layers: [{size: fqbins * 4, init: 'orthoUniform', id: 'z_dev', depth: 1}, {size:z, depth:4, activation: 'elu'}]})

var decode_layers = [{size: binSize, depth: 4}]
var decoder = rnn({input_shape: [batchSize, z], layers: decode_layers})

//var discriminator = dense({input_shape:[batchSize, z], layers:})

var rate = .002
var optimizer = tf.train.adam(rate)
// run it
load_and_run()

async function load_and_run(){
//  await mnist.loadData()
  train()
  test()
}

function feed_fwd(input, train, size){

  let {mag, phase} = transform(input)
  let m = z_mean.flow(mag, train)
  let d = z_dev.flow(mag, train) 

  // sample from mean and deviation
  let sample = $.initializers.randomNormal({shape: m.shape, trainable: false}).mul(d).add(m)

  //sample = tf.unstack(sample).map(e => time.matMul(e.mul(filter.layer)))

  var decode = decoder.flow(sample, train)//.reshape([].concat([size || batchSize], [manifold, binSize, 1]))

  // decode is the filter
  // but need to unstack here and do this to all samples in batch
  var result = tf.stack(tf.unstack(decode).map(e => {
    return tf.sum(tf.sin(tf.expandDims(e, 0).transpose().matMul(ttime)), 0)
  }))
  return {result,  m, d, sample}
}


function train(){
  for(var x = 0; x < epochas; x++){
      tf.tidy(() => {
        var batch = Array(1).fill(0).map(e => nextBatch(binSize))
        let signal = tf.squeeze(tf.stack(batch.map(e => e.signal), 1))
 //       let time = tf.stack(batch.map(e => e.time), null)
        //console.log(tf.concat(rolled, [0]))
        //console.log(signal)
        //process.exit()
        var _loss = $.scalar(0)
        var dispose = [signal, time]
        _loss = _loss.add(optimizer.minimize(function(){
          var {result, m, d, sample} = feed_fwd(signal, true)
  //        result = result.mul(time)
          //result = tf.unstack(result.slice([0,0], [1]))[0]//.map(e => e.squeeze()) // audio channel only
          var reconLoss = tf.mean(tf.losses.softmaxCrossEntropy(signal, result))// tf.sqrt(tf.mean(tf.square(tf.sub(signal, result))))//tf.losses.meanSquaredError(signal, res)
          var KL_loss = tf.sum($.scalar(1).add(d).sub(tf.square(m)).sub(tf.exp(d)),-1).mul($.scalar(-.5))
          var totes = tf.mean(KL_loss.add(reconLoss))//.add(regen)
          //let reg = [z_mean, z_dev, decoder].reduce((reg, topo) => reg.add(topo.regularize()), $.scalar(0))
            //console.log(`current regularario  is: ${reg.dataSync()}`)
            console.log(`current reconstruct loss is: ${reconLoss.dataSync()}`)
            console.log(`current kl loss is: ${tf.mean(KL_loss).dataSync()}`)
          $.dispose([totes, m, d, KL_loss, sample])
          return totes//.add(reg)
        }, true))
 //     console.log('\n********************************************')
 //     console.log(`tf memory is ${JSON.stringify(tf.memory())}`)
   //   console.log(`average loss after epoch ${(x+1)}:`) 
     // console.log(_loss.div($.scalar(1)).dataSync()[0] + '\n')
      $.dispose(dispose, true)
      dispose = []//encoder.disposal.map(e => false).filter(Boolean)
    })
  }
}

function test(input){
 // TODO: update for node backend 
  // reconstruct a few digits
  try{
  filter.save()
  z_mean.save()
  z_dev.save()
  decoder.save()
  }catch(err){console.log(err)}
  var keke = require('fs').createWriteStream('./kekekeke2.raw')
  for(var x = 0; x < 3000; x++){
    tf.tidy(() => {
        var batch = Array(1).fill(0).map(e => nextBatch(binSize))
        let signal = tf.squeeze(tf.stack(batch.map(e => e.signal)))
      var {result, m, d} = feed_fwd(signal, true) 
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


