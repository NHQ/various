async function* space(){
  let pp = new Promise(res => {
    document.addEventListener('keydown', e => res())
  })
  for await (x of pp) { yield {hello: 'world' }}
} 
async function main () {
  for await (const _ of (async function* () {})()) {
  }
}
async function yolo(){
  for await (const lost of space()) {
    continue
  }
}
