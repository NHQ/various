const $ = require('./utils')
const tf = $.tf

module.exports = dense

function dense({input_shape, layers}){

  var lastOutput = input_shape[1] 
  var variables = [] 
  var rootOp = function(input){return input}

  var flow = layers.reduce((a, config) => {
    let shape = [lastOutput, config.size] 
    let layer = tf.variable($.randomNormal({shape}))
    variables.push(layer)
    lastOutput = config.size 
    let fn = a
    let activation = tf[config.activation] || rootOp
    return function(input){
      let output = fn(input)
      return activation(output.matMul(layer))
    }}, rootOp)    

  var outputShape = [input_shape[0], lastOutput]
  return {flow, variables, outputShape}

}

