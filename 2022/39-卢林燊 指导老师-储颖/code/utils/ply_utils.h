//
// Created by lj on 2022/11/9.
//

#ifndef FUXIAN_PLY_UTILS_H
#define FUXIAN_PLY_UTILS_H


class PlyUtils {
public:
    static float calRadianOfPoints(const float *d1, const float *d2);

    static float radian2angle(float radian);

    template<class T1, class T2>
    static float calDistanceForPoint(T1& t1, T2& t2);
};

template<class T1, class T2>
float PlyUtils::calDistanceForPoint(T1 &t1, T2 &t2) {
    return sqrt(pow(t1.getPos()[0]-t2.getPos()[0], 2) + pow(t1.getPos()[1]-t2.getPos()[1], 2) + pow(t1.getPos()[2]-t2.getPos()[2], 2));
}
#endif //FUXIAN_PLY_UTILS_H
