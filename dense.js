const $ = require('./utils')
const tf = $.tf

module.exports = dense

function dense({input_shape, layers, input=undefined, ortho=false}){
  $.assert(arguments['0'], ['input_shape', 'layers'])

  var lastOutput = input_shape[1] 
  var variables = [] 
  var rootOp = function(input){return input}
  if(input){
    rootOp = input.flow
    input_shape = input.outputShape
  }
  var flow = layers.reduce((a, e) => {
    let config = e
    config.shape = [lastOutput, config.size] 
    
    let {layer, activation} = $.variable(config)
    console.log(layer.size)
    variables.push(layer)
    lastOutput = config.size 
    
    let fn = a
    
    return function(input){
      let output = fn(input)
      return activation(output.matMul(layer))
    }}, rootOp)    

  var outputShape = [input_shape[0], lastOutput]
  return {flow, variables, outputShape}

}

