#ifndef SAMPLING_H
#define SAMPLING_H
#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <time.h>
#include <utility>

class Sampling {
struct line;
struct point;
struct cell;
public:
	Sampling();
	void clear();
	void initilize();
	void DartThrowing();
	//compute polygonal cells 
	void initializePolygonVoid();
	void updatePolygonVoid(cell* cell_, int circle_idx, int void_idx, std::vector<std::pair<point, int>>& updated_void_polygon);
	void finetunePolygonVoid(cell* cell_, int circle_idx, std::vector<std::pair<point, int>>& updated_void_polygon);
	void computePolygonVoid();
	bool DartThrowingInVoid();
	bool computeIntersectSegAndCircle(line seg, point center, double radius, std::vector<point>& intersect_points);
	double calculatePointsDistance(point p1, point p2);
	double calculateLeastDistance(point center, line seg, point& least_p);
	double scalar_product(point v1, point v2);
	double cross_product(point v1, point v2);
	void doubleCopyPointVector(std::vector<point>source_v, std::vector<point>& target_v);
	void halfCopyPointVector(std::vector<std::pair<point, int>>source_v, std::vector<std::pair<point, int>>& target_v);
	//void halfCopyPointVector(std::vector<point>source_v, std::vector<point>& target_v);
	bool calculateCirclesIntersect(point center1, point center2, double radius, std::vector<point>& intersect_points);
	bool isPointInPolygon(point query_p, std::vector<std::pair<point, int>> polygon);
	void storeVoidPolygons(std::vector<std::pair<point, int>> void_polygon, std::vector<std::vector<std::pair<point, int>>>& polygons);
	void addPolygons(std::vector<std::vector<std::pair<point, int>>> source, std::vector<std::vector<std::pair<point, int>>>& target);
	double computePolygonArea(std::vector<std::pair<point, int>> polygon);
	void randomPointInPolygon(std::vector<std::vector<std::pair<point, int>>> polygons, double polygons_area, point& sample_p);
	int randomIndex(std::vector<double> cumsum);
	bool CheckneighborCell(int idx_1, int idx_2, point sample_p);

struct point {
	double x;
	double y;
	friend point operator - (point& a, point& b) {
		point res;
		res.x = a.x - b.x;
		res.y = a.y - b.y;
		return res;
	}
	friend point operator + (point& a, point& b) {
		point res;
		res.x = a.x + b.x;
		res.y = a.y + b.y;
		return res;
	}
	friend point operator * (double& t, point& a) {
		point res;
		res.x = t * a.x;
		res.y = t * a.y;
		return res;
	}
	friend point operator / (point& a, int& t) {
		point res;
		res.x = a.x / t;
		res.y = a.y / t;
		return res;
	}
	friend bool operator != (point& a, point& b) {
		if (a.x != b.x || a.y != b.y)
			return true;
		else
			return false;
	}
};
struct line {
	point p1;
	point p2;
};

struct circle {
	point center;
	double radius;
};

struct neighbor;
struct cell {
	int idx_1;   //
	int idx_2;   //
	bool sample = false;
	//bool update = false;
	struct neighbor* neighbor_;
	std::vector<std::pair<point, int>> void_polygon;   //int (-1,0,1,2;circle idx
	std::vector<std::vector<std::pair<point, int>>> polygons;
};

struct neighbor {
	int circle_num;
	std::vector<circle> neighbor_circles;
	std::vector<bool> totally_uncover;
};

public:
	int grid_num;
	double r;//least distance
	double grid_size;
	int max_miss_num = 400;   //400
	int max_miss_num2 = 100;
	std::vector<std::vector<bool>> flag;
	std::vector<std::vector<point>> sample_points;
	std::vector<std::vector<cell*>> all_cells;

	int sample_num_1 = 0;
	int miss_num = 0;
	int try_num_1 = 0;

	int all_sample_num = 0;
};

#endif //SAMPLING_H

