const $ = require('./utils.js')
const argv = require('minimist')(process.argv.slice(2))
const tf = $.tf

class Boltzman{
  constructor(input_size, hidden_size, rate=1e-3, momentum=.01, decay=1e-5, dev=.01){
    this.input_size = input_size
    this.hidden_size = hidden_size
    this.dev = dev
    this.l1 = $.scalar(0)
    this.decay = $.scalar(decay)
    this.momentum = $.scalar(momentum)
    this._shape = [input_size, hidden_size]
    let w = $.variable({
      shape: this._shape
      , dev: .5,
      id: `boltzmanBrain-i${input_size}-h${hidden_size}`
    })
    this.save = w.saver
    this.weights = w.layer 
    this.rate = $.scalar(rate)
    this.hbias = $.variable({
      shape: [1, hidden_size],
      init: 'zeros'
    }).layer
    this.vbias = $.variable({
      shape: [1, input_size],
      init: 'zeros'
    }).layer
  }
  potential(){
    var data = tf.ones([this.input_size, this.input_size])
    var w = tf.ones(this.weights.shape)
    return this.energy(data, w).mean().neg()
  }
  energy(data, w){
    let wx = tf.dot(data, w || this.weights).add(this.hbias)
    let vb = tf.dot(this.vbias, data)
    let ht = tf.sum(tf.log($.scalar(1).add(tf.exp(wx))), 1)
    return tf.neg(ht).sub(vb)
  }
  sample(){
    let oneway = this.activate(this.query($.variable({dev: this.err, shape: [this.input_size, this.input_size]}).layer).result).result
    let otrovia = this.activate(this.query(one).result).result

  }
  query(data, target){
    var prob = tf.sigmoid(tf.dot(data, this.weights).add(this.hbias))
    //prob.print()
    var result = prob.greaterEqual(target || $.variable({init:'randomUniform', shape: [data.shape[0], this._shape[1]]}).layer).asType('float32')
    return {result, prob}
  }
  activate(data, target){
    var prob = tf.sigmoid(tf.dot(data, this.weights.transpose()).add(this.vbias))
    //prob.print()
    var result = prob.greaterEqual(target || $.variable({init:'randomUniform', shape: [data.shape[0], this._shape[0]]}).layer).asType('float32')
    return {result, prob} 
  }
  trainb(hidden){
    var act= this.activate(hidden.prob)
    var nas = tf.dot(act.result, query.result)

    var cost = tf.mean(this.energy(data)).sub(tf.mean(this.energy(act.result)))
    var error = tf.sum(data.sub(act.prob).pow($.scalar(2)))//.div($.scalar(this.input)).add(this.decay) // smol number
    var grad
    this.weights = tf.variable(this.weights.add(grad = this.rate.mul(pas.sub(nas).div($.scalar(data.shape[0])))).sub(this.decay.mul(cost)))
    this.rate = this.momentum.mul(this.rate).sub(grad)
    //this.rate = tf.variable(this.rate.div($.scalar(1).sub(this.momentum.pow(tf.log(error)).add($.scalar(1e-8)))))
    // need asymptotic momentum    1

    return {result: act.result, error, cost, rate: this.rate}
  }
  trainp(data, target){
    //var dirty =  target || data.add($.variable({dev: this.dev, shape: data.shape}).layer)
    //var fauxpas = this.query(dirty)

    var query = this.query(data)
    var h = this.activate(query.result)
    var q2 = this.query(h.result).prob

    var pas = tf.dot(query.result.transpose(), data)
    var nas = tf.dot(q2.transpose(), h.prob)

    var cost = tf.mean(this.energy(data)).sub(tf.mean(this.energy(h.result)))
    var error = tf.sum(data.sub(h.prob).pow($.scalar(2)))//.div($.scalar(this.input)).add(this.decay) // smol number
    var grad
    this.weights = tf.variable(this.weights.add(grad = this.rate.mul(pas.sub(nas).transpose().div($.scalar(data.shape[0])))).sub(this.decay.mul(cost)))
    this.rate = tf.variable(this.momentum.mul(this.rate).sub(grad))
    //this.rate = tf.variable(this.rate.div($.scalar(1).sub(this.momentum.pow(tf.log(error)).add($.scalar(1e-8)))))
    // need asymptotic momentum    1

    return {result: h.result, error, cost, rate: this.rate}
  }
  traind(data, target){
    var dirty = target || data.add($.variable({dev: this.dev, shape: data.shape}).layer)
    var fauxpas = this.query(dirty)

    var query = this.query(data)
    
    var act= this.activate(query.result)
    var pas = tf.dot(fauxpas.result.transpose(), dirty)
    var nas = tf.dot(query.prob.transpose(), act.prob)
    //this.energy(act.result).mean().print()
//process.exit()
    var cost = tf.mean(this.energy(data)).sub(tf.mean(this.energy(act.result)))
    var error = tf.sum(data.sub(act.prob).pow($.scalar(2)))//.div($.scalar(this.input)).add(this.decay) // smol number
    var grad
    this.weights = tf.variable(this.weights.add(grad = this.rate.mul(pas.sub(nas).transpose().div($.scalar(data.shape[0])))).sub(this.decay.mul(cost)))
    this.rate = tf.variable(this.momentum.mul(this.rate).sub(grad))
     
    //this.weights = tf.variable(this.weights.add(this.rate.mul(pas.sub(nas).transpose().div($.scalar(data.shape[0])))).sub(this.decay.mul(cost)))
    //this.rate = tf.variable(this.rate.div($.scalar(1).sub(this.momentum.pow(tf.log(error)).add($.scalar(1e-8)))))
    // need asymptotic momentum    1

    return {result: act.result, error, cost, rate: this.rate}
  }
}

module.exports = Boltzman
//Test()
demnist()
async function demnist(){
  let mnist = require('./data.js')
  var chalk = require('chalk')
  var hft = require('../audio/fft/hft')
  var ndarray = require('ndarray')
  console.vlog = _ => console.log(_.split('').map(e => Number(e) === 0 ? chalk.black.bgBlue('0') : chalk.black.bgGreen('0')).join(''))
  await mnist.loadData()
  let size = 784
  let epochas = Number(argv.e) || 0
  let embed = 4
  let data = mnist.nextTrainBatch(size)
  let image = data.image.reshape([size, size]).greaterEqual($.scalar(.6785)).cast('float32')
  let roll = $.createRollMatrix(28*embed, 1)
  let label = dramadah(data.label.concat(tf.zeros([size, 28 * embed - 10]),1)).concat(tf.zeros([size, size-28*embed]), 1)
  let onez = data.label.concat(tf.ones([size, 28 * embed - 10]),1).concat(tf.zeros([size, size-28*embed]), 1)
  var codex = image.add(label)
  var bm = new Boltzman(size, size*4, 1e-2,  .9, 1e-6, .1)

  //console.log(codex.dataSync().slice(0,size).join(''))
  //tf.unstack(codex).forEach(e => tf.unstack(e.reshape([28,28])).forEach(e => console.vlog(e.dataSync().join(''))))
  //console.log(data.label.dataSync().slice(0,28).join(''))
  //process.exit()
  var res
  for(var i = 0; i < epochas; i++)
    tf.tidy(_ => {
      res = bm.traind(image, codex)//, target)
    tf.unstack(res.result).slice(0,2).forEach(e => tf.unstack(e.reshape([28,28])).forEach(e => console.vlog(e.dataSync().join(''))))
      console.log(i)
      res.cost.print()
      res.error.print()
      if(argv.s) bm.save()
      //res.rate.mean().print()
      //bm.decay.mul(res.cost).print()
    })
  tf.tidy(_=>{
    tf.unstack(bm.activate(bm.query(image).result).result).forEach(e => tf.unstack(e.reshape([28,28])).forEach(e => console.vlog(e.dataSync().join(''))))
  })
  tf.tidy(_=> {
    res = bm.trainp(codex)//, target)
    res.cost.print()
    res.error.print()
    res.rate.mean().print()
    bm.decay.mul(res.cost).print()
  }) 
    function dramadah(tensor){
    let stack = tf.unstack(tensor)
    return tf.stack(stack.map(vector => {
      let b = ndarray(new Int32Array(vector.size))
      let nd = ndarray(vector.dataSync())
      hft(nd,  b)
      return tf.tensor(b.data, vector.shape)
    })).reshape(tensor.shape)
  }

}



function Test(){

  var hft = require('../audio/fft/hft')
  var ndarray = require('ndarray')
  var chalk = require('chalk')
  console.vlog = _ => console.log(_.split('').map(e => Number(e) === 0 ? chalk.black.bgBlue('0') : chalk.black.bgGreen('0')).join(''))
  var size = 32
  const err = .1
  const epochas = 32 * 2
  // the energy and decay are correlated, therefor decay may be correlated to the gaussian energy potential...?
  // error and rate are correlated therefor rate may correlate to... 
  const bm = new Boltzman(size, 5120/4, 1e-3, .9, 1e-5, err)
  
  console.log(bm.potential().dataSync())
  var td = Array(size).fill(0).map((e,i) => {
    let b = ndarray(new Int32Array(size))
    let n = $.variable({shape: [1, size], init: 'zeros', max: .67}).layer.round().relu().asType('int32')
    let nd = ndarray(n.dataSync())
    let target = ndarray(n.dataSync())
    if(i < size/2){
      nd.set(i, 1)
      target.set(i, 1)
    }
    else{
      nd.set(i-16, 1)
      target.set(i-16, 1)
      nd.set(i-14, 1)
      target.set(i-14, 1)
    }
    hft(nd,  b)
    return {input: tf.tensor(b.data, [1,size]), target: tf.tensor(target.data, [1, size])}
  })
  td.forEach(e => console.vlog(e.input.dataSync().join('')))//.print())
  console.log('###################')
  td.forEach(e => console.vlog(e.target.dataSync().join('')))//.print())
  console.log('###################')
  //process.exit()

  //process.exit()
  var training_data = tf.squeeze(tf.stack(td.map(e => e.input)))
  var target_data = tf.squeeze(tf.stack(td.map(e => e.target)))
  for(var i = 0; i < epochas; i++)
    tf.tidy(_=> test(bm, training_data, null))//target_data))


  //td.forEach(e => console.log(e.dataSync().join('')))//.print())
  console.log('###################')
  td.forEach(e => {
    //console.log(e.add($.variable({dev: .2, shape: e.shape}).layer).round().dataSync().join(''))//print()
    //console.log(bm.activate(bm.query(e).result).result.dataSync().join(''))//print()
  })
  td.forEach(e => {
    console.vlog(bm.activate(bm.query(e.input.add($.variable({dev: err, shape: e.input.shape}).layer)).result).result.dataSync().join(''))//print()
    //console.vlog(bm.query(e.input.add($.variable({dev: err, shape: e.input.shape}).layer)).result.dataSync().join(''))//print()
    //console.log(bm.activate(bm.query(e).result).result.dataSync().join(''))//print()
  })
  td.forEach(e => {
        console.vlog(bm.activate(bm.query(e.input).result).result.dataSync().join(''))//print()
  })

  
  bm.activate(bm.query(training_data.mul($.scalar(2/4))).result).result.unstack().forEach(e => console.vlog(e.dataSync().join('')))
  

  function test(bm, data, target){
    var res = bm.trainp(data.mul($.scalar(3/4)))//, target)
    //res.result.print()
      res.cost.print()
      res.error.print()
      res.rate.mean().print()
      bm.decay.mul(res.cost).print()
    if(false && res.cost.abs().dataSync()[0] < 1){
      training_data.forEach(e => {
        console.log(bm.activate(bm.query(e.mul($.scalar(8)).add($.variable({dev: .1, shape: e.shape}).layer)).result).result.dataSync().join(''))//print()
      })
      console.log('*************************')
    }
    //res.error.print()
  }
}
