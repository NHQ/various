# unconventional-neighbours

A little module for generating unconventional neighborhoods (axis only, corners only, edges only and faces only) of arbitrary range and dimensions.

Inspired by and API-compatible with the [moore](https://www.npmjs.com/package/moore) module (see also [von-neumann](https://www.npmjs.com/package/von-neumann)).

## Installation

```
npm install unconventional-neighbours
```

## Usage

```js
// basic require
var neighbours = require('unconventional-neighbours');

neighbours.axis(range, dimensions);
neighbours.corner(range, dimensions);
neighbours.edge(range, dimensions);
neighbours.face(range, dimensions);
```

```js
// deep requires for optimized browserified package
var axis = require('unconventional-neighbours/functions/axis'),
    corner = require('unconventional-neighbours/functions/corner'),
    edge = require('unconventional-neighbours/functions/edge'),
    face = require('unconventional-neighbours/functions/face');

axis(range, dimensions);
corner(range, dimensions);
edge(range, dimensions);
face(range, dimensions);
```

Each function takes two arguments and returns an array of relative coordinates.

* `range` determines how large the neighborhood extends, and defaults to 1.
* `dimensions` determines how many dimensions the neighborhood
  covers - i.e. 2 will return the results for a 2D grid, and 3 will return the
  results for a 3D grid. May be any value above zero.
