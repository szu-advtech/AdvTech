//
// Created by lj on 2022/11/6.
//

#ifndef FUXIAN_POINT_H
#define FUXIAN_POINT_H


class Point {
public:
    float *_pos;
    float *_direction;
    Point();
    Point(float x, float y, float z, float nx, float ny, float nz);
    Point(float *pos, float *direction);

    // 移动拷贝构造函数
    Point(Point &&point) noexcept;
    // 拷贝构造函数
    Point(const Point &point);
    // 析构函数
    ~Point();
    float* getPos();
    float* getDirection();
};


#endif //FUXIAN_POINT_H
