import os;
import util

if __name__ == '__main__':
    modeToAttVals1 = {};
    modeToAttVals1[0] = set();
    modeToAttVals1[0].add(1);
    modeToAttVals1[0].add(2);
    modeToAttVals1[0].add(3);
    modeToAttVals1[0].add(4);
    modeToAttVals2 = {};
    modeToAttVals2[0] = set();
    modeToAttVals2[0].add(3);
    modeToAttVals2[0].add(4);
    modeToAttVals2[0].add(5);
    modeToAttVals2[0].add(6);
    insec_dimes = modeToAttVals1[0] & modeToAttVals2[0];
    print(insec_dimes)
