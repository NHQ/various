# von-neumann

A little module for generating Von Neumann neighborhoods (i.e. the surrounding cells
of a single cell in a grid) of arbitrary range and dimensions.

Inspired by and API-compatible with the [moore](https://www.npmjs.com/package/moore) module.

## Installation ##

``` bash
npm install von-neumann
```

## Usage ##

### `require('von-neumann')(range, dimensions)` ###

Takes two arguments, returning an array of relative coordinates.

* `range` determines how large the neighborhood extends, and defaults to 1.
* `dimensions` determines how many dimensions the Von Neumann neighborhood
  covers - i.e. 2 will return the results for a 2D grid, and 3 will return the
  results for a 3D grid. May be any value above zero.

``` javascript
var vonNeumann = require('von-neumann')

// 2D, 1 range:
vonNeumann(1, 2) === [
           [ 0,-1],
  [-1, 0],          [ 1, 0],
           [ 0, 1],
]
```

## Changelog

### 1.0.1 (2017-06-29) :

* Faster implementation by [Brandon Semilla](https://github.com/semibran)

### 1.0.0 (2015-09-20) :

* First implementation

## License

MIT
