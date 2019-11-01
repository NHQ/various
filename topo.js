const $ = require('./utils')
const tf = $.tf

module.exports = {dense, rnn, conv, iir}

const scalar_zero = $.scalar(0)//tf.variable(tf.scalar(0, 'float32'), false) 

function iir({input_shape, layers, depth=3, mag=.1, input=undefined, ortho=false, xav=true, cellfn=null}){
  let fwd = rnn(arguments[0])
  let fbk = rnn(arguments[0])
  let flow = function(input, train=true){
    return fwd.flow(input, 'fwd', train).add(fbk.flow(input, 'fbk', train))
  }
  return {flow}
}

function rnn({input_shape, layers, depth=3, mag=.1, input=undefined, ortho=false, xav=true, cellfn=null}){
  $.assert(arguments['0'], ['input_shape', 'layers'])
  
  var lastOutput = input_shape[1] 
  var variables = [] 
  var disposal = []
  var rootOp = function(input){return input}
  if(input){
    rootOp = input.flow
    input_shape = input.outputShape
  }
  function regularize(){
    return variables.reduce((a, e) => tf.add($.regularize({input: e, l:.001, ll:.001}), a), $.scalar(0)).mul($.scalar(1/input_shape[0]))
  }
  function save(){
    variables.forEach(e=>e.save())
  }
  var flow = layers.reduce((a, e, i) => {
    let config = e
    config.shape = [lastOutput, config.size] 
    //  let scalar_mag = tf.variable(tf.scalar(mag, 'float32'), false) // ¿trainable?
    var feedback = new Array(depth).fill(0).map(e => tf.zeros([1, config.size])) 
    var fb_w = new Array(depth).fill(0).map(e => $.variable({dev: !(xav) ? 1 : 1 / lastOutput, id: config.id ? config.id + i:false, shape: [config.size, config.size], trainable: true}).layer ) 
    if(xav) config.dev = 1 / lastOutput
    let {layer, activation} = $.variable(config)
    variables.push(layer)
    variables = variables.concat(fb_w)
    lastOutput = config.size 
    
    let fn = a



    return function(input, direction='fbk', train=true){
      var output = activation(fn(input, train).matMul(layer))
      if(true){
        var fb = feedback.map((e,i) => activation(e.matMul(fb_w[i])))
        var prev = fb.reduce((a, e) => e.add(a), scalar_zero)
        //console.log(prev, output)
        //if(!(prev.shape[0] === output.shape[0])) prev = prev.slice([0,0], [input.shape[0], input.shape[1]])
        $.dispose([feedback[0]]) 
        feedback.shift()
        if(direction==='fwd'){
          if(train) feedback.push(tf.variable(output, false))
          output = output.add(prev.div($.scalar(depth)))
        }
        else{
          output = output.add(prev.div($.scalar(depth)))
          if(train) feedback.push(tf.variable(output, false))
        }
        if(cellfn){
          output = cellfn(output)
        }
        $.dispose([fb, prev])
      } // else if generate, sum the feedbacks to gen_count dimensions each 
      else{ // take mean of feedback and splash it onto n input samples
        var fb = feedback.map((e,i) => e.matMul(fb_w[i]))
        var prev = fb.reduce((a, e) => tf.mean(tf.sigmoid(e.add(a)), scalar_zero), 1)
        output = output.add(prev)
        $.dispose([feedback[0]]) 
        feedback.shift()
        if(cellfn){
          output = cellfn(output)
        }
        feedback.push(tf.variable(output, false))
        $.dispose(fb)
      }
      return output
    }}, rootOp)    

  var outputShape = [input_shape[0], lastOutput]
  return {flow, variables, outputShape, disposal, regularize, save}

}

function conv({input_shape, layers, input=undefined, xav=true}){
  $.assert(arguments['0'], ['input_shape', 'layers'])

  var lastOutput = input_shape[2] 
  var lastDepth = input_shape[3] || 1 
  var variables = [], saves = [] 
  var rootOp = function(input){return input}
  if(input){
    rootOp = input.flow
    input_shape = input.outputShape
  }

  function regularize(){
    variables = variables.map(e=> e.mul(variables.reduce((a, e) => tf.add($.regularize({input: e}), a), $.scalar(0))
  ))}
  function save(){
    saves.forEach((e,i)=>e())
  }
  

  var flow = layers.reduce((a, e) => {
    let config = e
    let size = config.size
    if(!Array.isArray(size)) size = [size, size] // square filter
    config.shape = [].concat(size, [lastDepth, config.depth || 1])
    if(xav) config.dev = 1 / lastDepth
    //tf.keep(layer)
    lastOutput = config.size 
    lastDepth = config.depth || 1
    let {filter, layer, pool, activation, saver} = $.conv2d(config) 
    variables.push(layer)
    saves.push(saver)
    let fn = a
    
    return function(input){
      let output = filter(fn(input))
      return output
    }}, rootOp)    

  var outputShape = [input_shape].concat([lastDepth])
  return {flow, variables, outputShape, regularize, save}
}


function dense({input_shape, layers, input=undefined, ortho=false, xav=true}){
  $.assert(arguments['0'], ['input_shape', 'layers'])

  var lastOutput = input_shape[1] 
  var variables = [], saves = [] 
  var rootOp = function(input){return input}
  if(input){
    rootOp = input.flow
    input_shape = input.outputShape
  }

  function regularize(){
    variables = variables.map(e => e.mul(variables.reduce((a, e) => tf.add($.regularize({input: e}), a), $.scalar(0))))
  }
  
  function save(){
    saves.forEach((e,i)=>e(variables[i]))
  }
  
  var flow = layers.reduce((a, e) => {
    let config = e
    config.shape = [lastOutput, config.size] 
    if(xav) config.dev = 1 / lastOutput
    let {layer, activation, saver} = $.variable(config)
    tf.keep(layer)
    saves.push(saver)
    variables.push(layer)
    lastOutput = config.size 
    
    let fn = a
    
    return function(input){
      let output = fn(input).matMul(layer)
      let result = activation(output)
      return result
    }}, rootOp)    

  var outputShape = [input_shape[0], lastOutput]
  return {flow, variables, outputShape, regularize, save}
}

function lsmt({input_shape, layers, depth=3, mag=.1, input=undefined, ortho=false, xav=true}){

  
  let cellfn = function(input){
    
  }

}

