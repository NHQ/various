this is a tensorflow.js app;  it is a "handwritten" dense network, currently attempting a simple sanity check, to get [these same results](https://stats.stackexchange.com/questions/190148/building-an-autoencoder-in-tensorflow-to-surpass-pca#answer-307746) 

status: failing;  current reconstruction loss is ~.245, with sigmoid activation, worse with elu 

losses, printed to the console, show the details;  w/e the case, results are not like the linked stackexchange, with .056 reconstrction error after only 2 epochs

updated to use node-gpu backend, and node-mnist 

clone the repo, install deps with npm, run /app.js
```
npm i
node app.js
```
