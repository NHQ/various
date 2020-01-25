const fs = require('fs')
const buffers = require('buffers')
const manifest = require('../data/cline/meta.json')
const $ = require('../utils.js')
const tf = $.tf
//console.log(tf.losses)
const {dense, cnn, rnn, iir} = require('../topo.js')
const sampleRate = manifest.sampleRate
const btoa = $.btoa//require('to-arraybuffer')
const atob = $.atob// require('to-arraybuffer')

var length = 1
var channel_out = 4
var size = 1024 
var batch_size = [channel_out, size]
//var filter = iir({input_shape: batch_size, layers})
var opt = tf.train.adam(.01, .97, 9)

var stack = _ => {
  var layers = [{size: size * length, depth: 4, activation: 'tanh'}]
  var filters = Array(channel_out).fill(0).map(e => iir({input_shape: batch_size, layers}))
  function flow(input, train) {
    return filters.map(e => e.flow(input, train))
  }
  function reg(){
    return filters.reduce((a, e) => e.regularize().add(a), $.scalar(0))
  }
  function fuzz(){
    filters.forEach(e => e.fuzz())
  }
  return {flow, reg, fuzz}
}

var model = stack()

var tracks = {multi: new Array(channel_out).fill(new buffers), mono: new buffers}

var track = new Float32Array(btoa(fs.readFileSync('../data/cline/track.raw')))

for(var i=0; i<Math.floor(track.length/size); i++){
  let cut = track.slice(i*size, size+i*size)
  let input = tf.tensor(cut, [1, size * length])
  train(input, -1.45, (result, reset) =>{
    if(reset){
      $.dispose([result.mono, result.multi, input], true)
      model.fuzz()
      //opt = tf.train.adam(.00001, .67, 9)
      i--
    }
    else{
      tracks.mono.push(atob(result.mono.dataSync()))
      for(var j = 0; j < channel_out; j++){
        tracks.multi[j].push(atob(result.multi[j].dataSync()))
      }
      $.dispose([result.mono, result.multi, input], true)
    }
  })
  //$.dispose([result], false)
  //tracks.multi.forEach((e,i) => e.push(multi[i]))
  //tracks.mono.push(mono)
}

fs.writeFileSync('./combined.raw', tracks.mono.toBuffer())

tracks.multi.forEach((e,i)=>{
  fs.writeFileSync('./mono_channel_'+i+'.raw', tracks.multi[i].toBuffer())
})

function ad(x, y){
  return $.scalar(1).sub(tf.acos(x.mul(y).sum().div(tf.sqrt(x.pow($.scalar(2)).sum()).mul(tf.sqrt(y.pow($.scalar(2)).sum())))).div($.scalar(Math.PI)))
}

function train(input, threshold=.1, cb){
  var loss = 1
  var epoch = 0
  //let batch = Array(channel_out-1).fill(0).reduce(a => tf.concat([a, input], 0), input)
  var rutted = false, rutt = 0
  var losses = []
  var latest 
  var reset=false
  var channels = {mono: null, multi: null}
    tf.tidy(_=>{
      //$.dispose([], true)
      opt.minimize(_ =>{
        let result = model.flow(input, true)//.add($.scalar(1e-5))
        let mono = tf.stack(result).sum(0).div($.scalar(4))
        //let l= tf.abs(input.mul(mono).sum(1)).add(tf.losses.absoluteDifference(input, mono)).squeeze()
        let p = ($.pearson(input, mono))//.dataSync()[0]
        let a = ad(input, mono)//tf.neg(ad(input, mono).add(p))//.dataSync()[0]
        loss = a.dataSync()[0]
        console.log(loss)//,p.dataSync()[0])
        if(isNaN(loss)) mono.print()//reset = true
        epoch++
        $.dispose([channels.mono, channels.multi], false)
        channels.mono = tf.variable(mono)
        channels.multi = result.map(e => tf.variable(e))
        $.dispose([p,mono], false)
        $.dispose(result, false)
        return a//.add(model.reg())//.neg()
      }, true)//.dataSync()[0]
    })
    
      let mem = tf.memory()
      //console.log(mem.numBytes, mem.numTensors)
      cb(channels, reset)
}
