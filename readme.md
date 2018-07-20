experimental deep learning from scratch, with javascript, node.js and tensorflow

main work going into `topo.js` and `utils.js`

current experiemts are with mnist, but that's gotten old quickly

results:  we've seen > 98% classification, and various crappy reconstructions, but neither has been the goal

the goal has been to do build fundamentals in functional and expressive style, so we can innovate

recently added to training variables: xavier initialization, regularization;  this was good

soon:  adding more topologies, innovating

to see testing in action, clone the repo, install deps with npm, run /app.js
```
npm i
node app.js
```

it should run in the browser (see package.json), but it won't be webGL unless line 4 of utils.js is commented out

