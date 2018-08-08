var fs = require('fs')
var bta = require('buffer-to-typedarray')
var jbuff = require('buffers')
var walk = require('walker')
var $ = require('./utils')

/*  flow: load bin size per each beat per each song
    loadNext returns next bin size
    if index + bin size > beat length, go into next beat
    if last beat, wrap around to beginning... so need total lengths, and beat lengths
    or put all the buffers together with jbuffers
*/

function load(path='.'){

  var dirs = {}

  walk(path).on('directory', function(dir, stat){
    dirs[dir] = {}
    var meta = require(dir + '/meta.json')
    dirs[dir].meta = meta
    
  })
  let dk = Object.keys(dirs)
  return function(batchSize=dk.length, binSize=256){
    let batch = []
    while(batch.length < batchSize){
      let b = dk[Math.floor(Math.rabdon() * dk.length)]
      if(!batch.includes(b)) batch.push(dirs[b])
    }
   
    index = 0
    bindex = 0

    batch = batch.map(meta => {
      meta.fd = [] 
      meta.files.forEach((e, i) => {
        let d = {start: meta.times[i] - (meta.times[i-1] || 0), end: meta.times[i]} 
        d.fd = fs.openSync(e, 'r')
        d.stat = fs.statSync(e)
        meta.fd.push(d)
        //meta.fd[e].buffer = fs.readFileSync(e)
      })
    })
    function nextBatch(){
        
    }
    return nextBatch 
  }


  /* TODO perhaps load all samples if one song only, then read over them cicularly % bin size 
  */


  function loadNext(cb){
    fs.readFile(meta.files[index++%meta.files.length], (e, r) => e ? cb(e, null) : cb(e, bta(r)))
  }

  function loadBatch(bin=256, cb){
    let samples = []
    bin = bin*4 // float32 only
    let buff = new Buffer(bin)
    fs.readSync((meta.fd[meta.files[index++%meta.files.length]], buff, 0, bin, bindex+bin))
    samples.push(bta(buff)) 
    bindex+=bin
    cb(null, $.tf.stack(samples.map(e => $.tf.tensor1d(e))))
  }

}
