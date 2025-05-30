import getPort from 'get-port'

function range(start, end) {
    const iterable = {};
    iterable[Symbol.iterator] = function() {
      let n = start;
      return {
        next() {
          const value = n;
          n++;
          return { value, done: value === end };
        }
      }
    }
    return iterable;
  }
  

export const findPort = async () => {
  // Start at 5 digits because people rarely use these.
  return getPort({ port: range(10000, 65535) });
};