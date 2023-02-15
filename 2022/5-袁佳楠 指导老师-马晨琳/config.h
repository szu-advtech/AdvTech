#ifndef _CONFIG_H_
#define _CONFIG_H_

namespace config {
  const int FANOUT = 30;
  const int BUFFER_SIZE = 30 + 1;
  double ERROR = 240;
  const int C = 50;
  const int PERFORMANCE_FIRST = 0;
  const int STORAGE_FIRST = 1;
}

enum State {
  SUCCESS,
  FAIL
};

#endif
