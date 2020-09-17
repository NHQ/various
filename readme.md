### ML framework in TensorFlow.js

This is my personal framework, written from base matrix operations. I wrote it to prove to myself I grok the concepts and Maths, and as a testing ground for my intuition and ideas, outside of the box of frameworks.  

Notable pieces and projects:
* [basic NNs with support for layers](topo.js#L32)
* [utils](utils.js) for initialization, and Maths / functions not included in TF
* Boltzman Machines
  * [Restricted BMs](boltzman.js) Class and Test
  * [experimental proof for distributed training](bfm.js#L148)
* [Experimental Neural IIR](topo.js#L10)
* [MNIST solution with CNNs](https://github.com/NHQ/mnist10k#mnist-10k)
* [Polynomial Regression to Find Bezier Control Points](http://nhq.github.io/beezy/public/)
  * Useful for ML assisted design and AI telemetry 
  * interactive demo
* [Tensor visualization](https://nhq.github.io/wavetable/public/) (Discreet Fourier Transforms over phase changes)
* [Streaming Video Batch Processing](https://github.com/NHQ/piped_convolution/blob/master/server.js)
  * Train and Test Batching
  * options: size, scale, frame start, and pixel formats
  * uses ffmpeg
