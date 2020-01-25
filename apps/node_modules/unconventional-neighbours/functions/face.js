module.exports = function face (range, dims) {
    "use strict";

    dims = dims || 2;
    range = range || 1;

    return recurse([], [], 0);

    function recurse (array, temp, d) {
        var i,
            k,
            match;

        if (d === dims-1) {
            for (i = -range; i <= range; i += 1) {
                match = (Math.abs(i) === range);
                for (k = 0; !match && k < dims; k++) {
                    match = (Math.abs(temp[k]) === range);
                }

                if (match) {
                    array.push(temp.concat(i));
                }
            }
        } else {
            for (i = -range; i <= range; i += 1) {
                recurse(array, temp.concat(i), d + 1);
            }
        }

        return array;
    }
};
