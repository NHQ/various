module.exports = function vonNeumann (range, dimensions) {
    "use strict";

    range = range || 1;
    dimensions = dimensions || 2;

    var size = range * 2 + 1,
        iterations = Math.pow(size, dimensions),
        center = (iterations - 1) / 2,
        neighbors = [];

    for (var i = 0; i < iterations; i++) {
        if (i !== center) {
            var neighbor = new Array(dimensions),
                distance = 0,
                remaining = i;

            for (var d = 0; d < dimensions; d++) {
                var remainder = remaining % Math.pow(size, d + 1),
                    value = remainder / Math.pow(size, d) - range;

                neighbor[d] = value;
                distance += Math.abs(value);
                remaining -= remainder;
            }

            if (distance <= range) {
                neighbors.push(neighbor);
            }
        }
    }

    return neighbors;
};
