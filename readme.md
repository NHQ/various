this is a tensorflow.js app;  it is a "handwritten" dense network, currently attempting a simple sanity check, to get [these same results](https://stats.stackexchange.com/questions/190148/building-an-autoencoder-in-tensorflow-to-surpass-pca#answer-307746) 

status: failing;  losses printed to the console every 500th iteration, all hover mysteriously around 0.478...

updated to use node-gpu backend, and node-mnist 

clone the repo, install deps with npm, run /app.js
```
npm i
node app.js
```
