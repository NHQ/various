var $ = require('./utils.js')
var tf = $.tf

var nextBatch = require('./load.js')()

var batch_size =1 //256 * 4
var epochas = 1.00000
var binSize = 1e3 // 128th of 48K 
var manifold = 1
var dimension = 1//2
var input_shape = [batch_size, manifold, binSize, dimension]
var z = binSize / 2 
var sampleRate = 16e9//Math.pow(2,34) // 2 billions
var range = [1e9, 2e9]
var offset = 3e4
var ff = 8e7//*21/13///2e9/2-1//Math.pow(2, 30) - 1e15// limit 
var f = ff//21/13//ff//2
var ph = 1/4//2//1/4
var gen = (e, i)=> Math.sin(Math.PI * 2 * ff * ((i + offset + sampleRate / ff * ph )/ sampleRate)) 

function write({freqs, sampleRate, duration, offset, wave}){
  freqs = freqs.map()
}

var signal = new Float32Array(binSize).fill(0).map(gen)
var asignal = new Array(binSize).fill(0).map(gen)

var mag = (a)=> Math.sqrt(Math.pow(a[0], 2) + Math.pow(a[1], 2))
var phase = (a)=> Math.atan2(a[0], a[1])//Math.sqrt(Math.pow(a[0], 2) + Math.pow(a[1], 2))
dd = $.jsdft(asignal, f, sampleRate)
dd = dd.reduce((a, e)=> [a[0]+e[0], a[1]+e[1]], [0,0])
console.log(dd, mag(dd)*2/z, phase(dd), Math.abs(phase(dd)/Math.PI/2))

signal = tf.tensor(signal, [1, binSize], 'float32')
console.log("signal energy", signal.square().mean().dataSync())
var filter = tf.linspace(0, z-1, z).reshape([1, z])
filter = $.variable({init: 'harmonic', base: f, size: 100, shape:[1, 100], trainable:false}).layer
var time = bin => tf.tensor(Array(z).fill(0).map((e,i)=> i / sampleRate), [1, z])//, z, z).div($.scalar(sampleRate))

var t = time(0).reshape([z, 1])
t.print()
t = $.tautime(binSize, sampleRate)//t.mul($.scalar(Math.PI * 2)) //tf.tensor([Math.PI * 2], [1,1]))

var d = $.dft(t, filter, sampleRate)

sin = d.sin(signal)
cos = d.cos(signal)
cos.print()
sin.print()

dft = $.mag(cos, sin).mul($.scalar(2)).div($.scalar(binSize))
dmag = dft.dataSync()
dph = $.phase(cos, sin).dataSync()

console.log(dmag[0], dph[0], 1-Math.abs(dph[0] / Math.PI%1) )
