module.exports = hft


function hft(signal, transform){
  if(signal.size === 1) return true
  var size = signal.size / 2
  var x = signal.hi(size) // first half
  var y = signal.lo(size) // second half
  var a = transform.hi(size)
  var b = transform.lo(size)
  for(var i = 0; i < size; i++){
    var xi = x.get(i) 
    var yi = y.get(i)
    var xory = xi ^ yi
    b.set(i, xory)
    a.set(i, xi)
  }
  hft(a, x)
  hft(b, y)
}

