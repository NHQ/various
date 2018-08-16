var fs = require('fs')
//var jbuff = require('buffers')
var walk = require('klaw-sync')

var $ = require('./utils')

/*  flow: load bin size per each beat per each song
    loadNext returns next bin size
    if index + bin size > beat length, go into next beat
    if last beat, wrap around to beginning... so need total lengths, and beat lengths
    or put all the buffers together with jbuffers
*/

let next = load('./data')()

for(var i = 0; i < 10000; i++) console.log(new Float32Array(next()[1])[255])


function load(path='.', cb){

  var tree = {}

  var dirs = walk(path, {nofile: true})

  dirs.forEach(dir => tree[dir.path] = require(dir.path + '/meta.json'))
  dirs = dirs.map(e => e.path)

  return function(batchSize=dirs.length, binSize=256){
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
    function nextBatch(binSize=256){
      //  need to go thru batch and return bin size at offset, push offset
      //  check fd has size remaining for bin, if not, get the most and move to the next fd
      return batch.map(e => {
        f = e.fd[0]
        let buf = new Buffer(binSize * 4)
        var read = fs.readSync(f.fd, buf, 0, binSize*4, f.index) 
        f.index += read
        if(read < binSize){
          f.index = 0
          e.fd.shift()
          e.fd.push(f)
          f = e.fd[0]
          read = fs.readSync(f.fd, buf, read, (binSize-read) * 4, f.index) 
          f.index += read 
        }
        return buf.buffer
      })    
    }
    return nextBatch 
  }
}
