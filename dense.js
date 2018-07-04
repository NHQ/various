const $ = require('./utils')
const tf = $.tf

module.exports = dense

function dense({input_shape, layers}){

  var lastOutput = input_shape[1] 
  var variables = [] 
  // this reduce op returns a function for each value in layers
  // eachfunction calls the layer above it
  // the initial function returns the input
  var flow = layers.reduce((a, config) => {
    let shape = [lastOutput, config.size] 
    let layer = tf.variable($.randomNormal({shape}))
    variables.push(layer)
    lastOutput = config.size 
    let fn = a
    let activation = tf[config.activation] || function(x){ return x}
    return function(input){
      let output = fn(input)
      return activation(output.matMul(layer))
    }}, function(input){
      return input
  })    

  var outputShape = [input_shape[0], lastOutput]
  return {flow, variables, outputShape}

}

