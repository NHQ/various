module.exports = function axis (range, dims) {
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
                match = (i === 0 ? 1 : 0);
                for (k = 0; k < dims; k++) {
                    match+= (temp[k] === 0 ? 1 : 0);
                }

                if (match === dims-1) {
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
