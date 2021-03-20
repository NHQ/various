var $ = exports
$.westerns = require('../westerns')
$.oz = require('../oscillators')
$.ph = require('../phasers')
$.amod = require('amod')
$.env = $.nvelope = require('../nvelope')
$.jsync = require('jsynth-sync')
$.zerone = require('../zerone')
$.jdelay = require('jdelay')
$.chrono = require('../jigger')
$.meffisto = require('../meffisto')
$.zerone = require('../zerone')
$.euclid = require('../euclid-time')
$.beatmath = require('beatmath')
$.teoria = require('teoria')
$.gtone = require('../gtones')
$.sigdelay = require('../new-deal')
$.dataDelay = require('../data-delay')
$.winfunk = require('../winfunk')
$.midi = require('web-midi')
$.beezxy = require('../beezy/beezy')
$.tnorm = require('../normalize-time')
$.beezmod = function(){
  let pts = [[0, .5], [.5, 1], [.5, .5], [.5, 0], [1, .5]]
  let env = $.beezxy(pts)
  pts = pts.reduce((a,e) => { a[0].push(e[0]); a[1].push(e[1]); return a }, [[],[]])
  return function(c, t, f){
    pts[0][2] = .5 + $.oz.sine(t, f) / 2
    return c * env((t * f) % 1, pts)[1]
  }
}

$.fract = function(v){
  return v - Math.floor(v)
}
$.quant = function(v, q){
  return Math.floor(v/q)*q  
}

$.sigmoid = function(t){
  return (1 / (1 + Math.pow(Math.E, -t))) * 2 - 1
}
$.sigmod = function(c, r, t, f){
  return c + (r * $.sigmoid($.amod(6, 6, t * 2, f))) 
}
$.alog = function(c, r, t, f){ return c + r * ((Math.log((1.0001 + $.oz.sine(t, f)) * 50) / Math.log(10))/2-2) }
$.ease = (pts, d, q=3)=> {
  let u = pts[2][1]
  let p = pts[1][1]
  let l = pts[0][1]
  let i = $.tnorm(0, d)
  return function (t){
    let tt = t
    t=t * (1/d)
    let e = ((Math.pow(t, 1/(p*q)) * (u-l)) + l)
    //console.log(tt, t, e, l, u, pts)
    return e
    //return e
  }
}


$.chain = function (f1, f2, d1, d2){
  var alt = [f1, f2] 
  return function(t){
    var e = Math.floor(Math.min(t/d1, 1)) // zero if t < d1, else 1
    return alt[e](t - (d1 * e))
  }
}

$.iir = function(d, c, fb=.5){
  d = d || 7
  c = c || 2
  var output = new Array(c)
  output.fill(0)
  output = output.map(function(e, i){
    return new Array(i+1).fill(0)
  })

  var delays = new Array(d)
  delays.fill(0)
  delays = delays.map(function(e, i){
    return new Array(i+1).fill(0)
  })
  return function(s, f=fb){
    let i = delays.reduce(function(aa, delay){
      var sample = delay[0]  
      delay.push(s)
      delay.shift()
      return sample + aa
    }, 0)  
    
    let j = output.reduce(function(aa, delay){
      var sample = delay[0] 
      delay.push(i)
      delay.shift()
      return sample + aa
    }, 0)  

    return j 
  }
}

