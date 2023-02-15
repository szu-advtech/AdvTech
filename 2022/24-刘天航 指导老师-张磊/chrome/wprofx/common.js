/* eslint-disable no-plusplus */
/* eslint-disable no-underscore-dangle */
/* eslint-disable no-prototype-builtins */
/* eslint-disable camelcase */
/* eslint-disable no-undef */
/* eslint-disable no-restricted-syntax */
module.exports = {
  convertIdtoHex(_trace) {
    for (trace_event in _trace) {
      if (_trace.hasOwnProperty(trace_event)) {
        if ('args' in trace_event && 'id' in trace_event && 'name' in trace_event && 'source_type' in trace_event.args) {
          // Convert the source event id to hex if one exists
          if ('params' in trace_event.args && 'source_dependency'
            in trace_event.args.params && 'id'
            in trace_event.args.params.source_dependency) {
            dependency_id = parseInt(trace_event.args.params.source_dependency.id, 10);
            trace_event.args.params.source_dependency.id = `0x${dependency_id}x`;
          }
        }
      }
    }
    return _trace;
  },

  isBalanced(_array) {
    const my_array = [..._array]; // cloned
    const _tmpStack = [];
    const _length = my_array.length;
    for (let j = 0; j < _length; j++) {
      if (my_array[j].ph === 'B') {
        _tmpStack.push('B');
      } else if (my_array[j].ph === 'E') {
        _tmpStack.pop();
      }
    }
    if (_tmpStack.length === 0) {
      return true;
    }
    return false;
  },

  mergeEvents(_array) {
    if (_array.length > 2) {
      return [[_array[0], _array.slice(-1)[0]], [_array.slice(1, -1)]];
      // return [[_array[0], _array[-1]], [_array[1:-1]]];
    }
    return [[_array[0], _array.slice(-1)[0]]];
    // return [[_array[0], _array[-1]]];
  },

  mapToJson(map) {
    return JSON.stringify([...map]);
  },

  getCol(matrix, col) {
    const column = [];
    for (let i = 0; i < matrix.length; i++) {
      column.push(matrix[i][col]);
    }
    return column;
  },
  round(value, decimals) {
    return Number(`${Math.round(`${value}e${decimals}`)}e-${decimals}`);
  },
};
