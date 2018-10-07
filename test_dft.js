var $ = require('./utils.js')
var tf = $.tf

var nextBatch = require('./load.js')()

var batch_size =1 //256 * 4
var epochas = 1.00000
var binSize = 16 // 128th of 48K 
var manifold = 1
var dimension = 1//2
var input_shape = [batch_size, manifold, binSize, dimension]
var z = 16
var sampleRate = 16
var f = 2

var gen = (e, i)=> Math.sin(Math.PI * 2 * f * i / sampleRate)

var signal = new Float32Array(binSize).fill(0).map(gen)
var asignal = new Array(binSize).fill(0).map(gen)

var mag = (a)=> Math.sqrt(Math.pow(a[0], 2) + Math.pow(a[1], 2))
dd = $.jsdft(asignal, f, sampleRate)
dd = dd.reduce((a, e)=> [a[0]+e[0], a[1]+e[1]], [0,0])
console.log(dd, mag(dd))

signal = tf.tensor(signal, [1, binSize], 'float32')
var filter = tf.tensor([f], [1,1])//$.harmonic(2, 16)
var time = bin => tf.tensor(Array(z).fill(0).map((e,i)=> i / sampleRate), [1, z])//, z, z).div($.scalar(sampleRate))

var t = time(0).reshape([z, 1])
t.print()
t = t.mul(tf.tensor([Math.PI * 2], [1,1]))

var d = $.dft(t, filter, sampleRate)

sin = d.sin(signal)
cos = d.cos(signal)
cos.print()
sin.print()

dft = $.mag(cos, sin)
console.log(dft.dataSync())
tf.linspace(0, .9375, 16).print()
