var net = require('net')
var $ = require('../utils.js')
var tf = $.tf
var {dense, rnn, conv, iir} = require('../topo.js')
var argv = require('minimist')(process.argv.slice(2))
var shape = [argv.b || 1, 3,3, 1] 
var state = $.variable({shape, init: 'randomUniform', trainable: false}).layer.round()
let game = $.gol(shape, state)
var batch_size = [argv.b || 1, shape[1] * shape[2]]
var stackSize = parseInt(argv.z) || 1
var headCount = parseInt(argv.h) || 1
var depth = parseInt(argv.d) || 1
var w_size = parseInt(argv.s) || shape[1] * shape[2]
var l_rate = parseFloat(argv.r)||.0001
var name = argv.n || ''
var drillDown = 1 
var optimus = tf.train.adam(l_rate)// .667, .9)
var {stack, save, regularize} = transformerer()
var stream = null
var pixels = new Uint8Array(shape[1] * shape[2])
var cnn = conv({input_shape: shape, layers: [{size: 3, depth: 1, activation: 'relu', pool: {fn: tf.avgPool, size: 3, strides: 1}}]})

train(game)

var server = net.createServer(stream=>{
  stream = stream
})

//server.listen(2233)


function train(game, epochas=argv.e || 10, futuras=argv.f || 3){// flatten the thing or what?  
  //var state = game.state
  for(var i = 0; i < epochas; i++){
    var batch = [], target = []
    for(var b =0; b < batch_size[0]; b++){
      let c = batch.push($.variable({shape: [1, shape[1], shape[2], 1], init: 'randomUniform', trainable: false}).layer.round())
      target.push(game.next(batch[c-1]).state)
    }
    batch = tf.stack(batch).reshape(shape)
    target = tf.stack(target).reshape(shape)
    for(var j = 0; j < futuras; j++){
      optimus.minimize(()=>{
        let c = cnn.flow(batch).reshape(batch_size)
        let output = stack(c).reshape(shape)
        let loss = tf.losses.softmaxCrossEntropy(target, output).sum()
        //console.log(output.dataSync())
        loss.print()
        let reg = regularize(.01, .001)
        //reg.print()
        //if(i % 10 == 0) output.print()
        //if(i % 10 == 0) target.print()
        return loss//.add(reg)
        //state.print()
      })
      batch = target//.reshape(shape)
      target = game.next(batch).state

    }
  }
  if(argv.save) save()
}

function transformerer(){
  const saves = []
  const regularizes = []
  const rootOp = e => e
  const stack = new Array(stackSize).fill(0).reduce((fn, e,i) => {
    var s = i
    var heads = new Array(headCount).fill(0).map((e, i) => {
      return {q: dense({input_shape: batch_size, layers:[{ size: w_size, depth: depth, id: `${name}s${s}qh${i}w${w_size}`} ]}) , 
              k: dense({input_shape: batch_size, layers:[{ size: w_size, depth: depth, id: `${name}s${s}kh${i}w${w_size}`} ]}) , 
              v: dense({input_shape: batch_size, layers:[{ size: w_size, depth: depth, id: `${name}s${s}vh${i}w${w_size}`} ]})
      }  
    })
    //var ff = dense({input_shape: [batch_size[0], w_size * headCount], layers: [{size: 1, activation: 'tanh', depth: depth}]})//, id: `s${s}h${i}w${w_size}`}]})
    var z = dense({input_shape: [batch_size[0], w_size * headCount], layers: [{size: w_size, activation: 'relu', depth: depth, id: `${name}s${s}z0w${w_size}`}]})
    
    let save = function(){
      heads.forEach(e => { 
        e.q.save()
        e.k.save()
        e.v.save()
      })
      z.save()
    }

    saves.push(save)

    let regularize = function(l=.01, ll=.01){
      return heads.reduce((a,e) => { 
        return e.q.regularize(l,ll).add(
        e.k.regularize(l,ll)).add(
        e.v.regularize(l,ll)).add(a)
      }, $.scalar(0)).add(
      z.regularize(l,ll))
    }

    regularizes.push(regularize)

    return function(input, train){
      input = fn(input)
      var qvc = heads.map((e, i) => {
        let q = e.q.flow(input, train)
        let v = e.v.flow(input, train)
        let k = e.k.flow(input, train)
        //console.log(q, v, k)
        let qk = k.transpose().matMul(q)
        let smx = $.tf.softmax(qk.div($.tf.sqrt($.scalar(shape[1]*shape[2]))))
        //console.log(qk, smx)
        let output = v.matMul(smx)
        return output
      })

      var q = tf.concat(qvc, 1)
      var output = z.flow(q, train)
      if(true || s < stackSize - 1) {
        return $.normalize(output.add(input))
      }
      else return output
    }
  }, rootOp)

  return {stack, save, regularize}

  function regularize(l=.01, ll=.01){
    return regularizes.reduce((a,e) => e(l, ll).add(a), $.scalar(0))
  }

  function save(){
    saves.forEach(e => e())
  }
}
