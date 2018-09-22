var fs = require('fs')
//var jbuff = require('buffers')
var walk = require('klaw-sync')

var $ = require('./utils')
var tf = $.tf
/*  flow: load bin size per each beat per each song
    loadNext returns next bin size
    if index + bin size > beat length, go into next beat
    if last beat, wrap around to beginning... so need total lengths, and beat lengths
    or put all the buffers together with jbuffers
*/

var test = false//true
if(test){
  let next = load('./data')

  for(var i = 0; i < 1; i++) {
    let {signal, time} = next() 
    signal.print()
    time.print()
  }
}

module.exports = load

function load(path='./data', batchSize){

  var tree = {}

  var dirs = walk(path, {nofile: true})

  dirs.forEach(dir => tree[dir.path] = require(dir.path + '/meta.json'))
  dirs = dirs.map(e => e.path)

  return (function(batchSize=dirs.length){
    let batch = []
    while(batch.length < batchSize){
      let b = dirs[Math.floor(Math.random() * dirs.length)]
      if(!batch.includes(tree[b])) batch.push(tree[b])
    }
   
    var index = 0
    var bindex = 0
    // batch of directory meta files, each directory full of chopped up samples
    // after this map, the meta will include fd's for to fs.read
    // possibly reduce to a function that returns tensor [batchSize, binSize]
    batch = batch.map(meta => {
      meta.fd = [] 
      meta.files.forEach((e, i) => {
        let d = {start: meta.onsets[i], end: meta.onsets[i+1] || 0} 
        d.fd = fs.openSync(e, 'r')
        d.stat = fs.statSync(e)
        d.index = 0
        d.length = d.stat.size / 4
        meta.fd.push(d)
      })
      return meta
    })
    //console.log(batch)
    function nextBatch(binSize=256, dims=2, bytes=4){
      //  need to go thru batch and return bin size at offset, push offset
      //  check fd has size remaining for bin, if not, get the most and move to the next fd
      let totes = binSize * bytes * dims
      let stack = tf.stack(batch.map(e => {
        
        f = e.fd[0]
        let buf = new Buffer(totes)
        var read = fs.readSync(f.fd, buf, 0, totes, f.index) 
        f.index += read
        if(read < totes){
          f.index = 0
          e.fd.shift()
          e.fd.push(f)
          f = e.fd[0]
          read = fs.readSync(f.fd, buf, read, totes-read, f.index) 
          f.index += read 
        }
        return tf.tensor(new Float32Array(buf.buffer), [1, binSize, dims])
      }))    
      $.dispose(stack)
      let signal = tf.squeeze(stack.slice([0,0,0,1]))//.reshape([-1, binSize])

      let time = tf.squeeze(stack.slice([0,0,0,0], [batch.length,1,binSize,1]))//.reshape([-1, binSize])

      return {signal, time}
    }
    return nextBatch 
  })(batchSize)
}
