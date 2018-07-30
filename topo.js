const $ = require('./utils')
const tf = $.tf

module.exports = {dense, rnn, conv}

const scalar_zero = tf.variable(tf.scalar(0, 'float32'), false) 

function rnn({input_shape, layers, depth=3, mag=.1, input=undefined, ortho=false, xav=true}){
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
  var flow = layers.reduce((a, e) => {
    let config = e
    config.shape = [lastOutput, config.size] 
    let scalar_mag = tf.variable(tf.scalar(mag, 'float32'), false) // ¿trainable?
    var feedback = new Array(depth).fill(0).map(e => tf.zeros([1, config.size])) 
    var fb_w = new Array(depth).fill(0).map(e => $.variable({dev: !(xav) ? 1 : 1 / lastOutput, shape: [config.size, config.size], trainable: true}).layer ) 
    if(xav) config.dev = 1 / lastOutput
    let {layer, activation} = $.variable(config)
    variables.push(layer)
    variables = variables.concat(fb_w)
    lastOutput = config.size 
    
    let fn = a

    
    return function(input, train=false){
      var output = fn(input, train).matMul(layer)
      if(train){
        var fb = feedback.map((e,i) => e.matMul(fb_w[i]))
        var prev = fb.reduce((a, e) => tf.sigmoid(e.add(a)), scalar_zero)
        output = output.add(prev)
        $.dispose([feedback[0]]) 
        feedback.shift()
        feedback.push(tf.variable(output, false))
        $.dispose(fb)
      } // else if generate, sum the feedbacks to gen_count dimensions each 
      return activation(output)
    }}, rootOp)    

  var outputShape = [input_shape[0], lastOutput]
  return {flow, variables, outputShape, disposal, regularize}

}

function conv({input_shape, layers, input=undefined, xav=true}){
  $.assert(arguments['0'], ['input_shape', 'layers'])

  var lastOutput = input_shape[1] 
  var lastDepth = input_shape[2] || 1 
  var variables = [] 
  var rootOp = function(input){return input}
  if(input){
    rootOp = input.flow
    input_shape = input.outputShape
  }

  function regularize(){
    return variables.reduce((a, e) => tf.add($.regularize({input: e}), a), $.scalar(0))
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
    let {filter, vars} = $.conv2d(config) 
    variables.push(vars.layer)
    let fn = a
    
    return function(input){
      let output = filter(fn(input)) 
      return output
    }}, rootOp)    

  var outputShape = [input_shape].concat([lastDepth])
  return {flow, variables, outputShape, regularize}
}


function dense({input_shape, layers, input=undefined, ortho=false, xav=true}){
  $.assert(arguments['0'], ['input_shape', 'layers'])

  var lastOutput = input_shape[1] 
  var variables = [] 
  var rootOp = function(input){return input}
  if(input){
    rootOp = input.flow
    input_shape = input.outputShape
  }

  function regularize(){
    return variables.reduce((a, e) => tf.add($.regularize({input: e}), a), $.scalar(0))
  }
  

  var flow = layers.reduce((a, e) => {
    let config = e
    config.shape = [lastOutput, config.size] 
    if(xav) config.dev = 1 / lastOutput
    let {layer, activation} = $.variable(config)
    tf.keep(layer)
    variables.push(layer)
    lastOutput = config.size 
    
    let fn = a
    
    return function(input){
      let output = fn(input).matMul(layer)
      let result = activation(output)
      return result
    }}, rootOp)    

  var outputShape = [input_shape[0], lastOutput]
  return {flow, variables, outputShape, regularize}
}

