#include "Sampling.h"

Sampling::Sampling()
{
}

void Sampling::clear() {
	flag.clear();
	sample_points.clear();
	all_cells.clear();
}

void Sampling::initilize() {
	//initilize flag vector
	for (int i = 0; i < grid_num; i++) {
		std::vector<bool> tmp;
		for (int j = 0; j < grid_num; j++) {
			tmp.push_back(false);
		}
		flag.push_back(tmp);
	}
	//initilize sample_points vector
	for (int i = 0; i < grid_num; i++) {
		std::vector<point> tmp;
		for (int j = 0; j < grid_num; j++) {
			point p1;
			p1.x = -1;
			p1.y = -1;
			tmp.push_back(p1);
		}
		sample_points.push_back(tmp);
	}
}

void Sampling::DartThrowing() {
	int iteration_num = 5 * grid_num * grid_num;
	//iteration_num = 10;

	std::default_random_engine e(time(0));
	std::default_random_engine e2(time(0));
	for (int i = 0; i < iteration_num; i++) {
		if (miss_num == max_miss_num)
			break;
		try_num_1++;

		//random choose one grid
		std::uniform_int_distribution<unsigned> u(0, grid_num-1);
		int random_1 = u(e);
		int random_2 = u(e);
		//std::cout << random_1 << " " << random_2 << std::endl;

		//random sample in grid
		double length = r / sqrt(2);
		std::uniform_real_distribution<double> u2(0, length);
		double x = u2(e2);
		double y = u2(e2);
		x = random_2 * length + x;
		y = random_1 * length + y;
		//std::cout << x << " " << y << std::endl;

		//Check neighboring grids
		bool miss_flag = false;
		for (int k = -2; k < 3; k++) {
			for (int l = -2; l < 3; l++) {
				int idx1 = random_1 + k;
				int idx2 = random_2 + l;
				if (idx1 >= 0 && idx1 < grid_num && idx2 >= 0 && idx2 < grid_num) {
					if (flag[idx1][idx2]) {
						double x_ = sample_points[idx1][idx2].x;
						double y_ = sample_points[idx1][idx2].y;
						double x_bias = x_ - x;
						double y_bias = y_ - y;
						double dis = sqrt(pow(x_bias, 2) + pow(y_bias, 2));
						if (dis < r) {
							miss_flag = true;
							break;
						}
					}
				}
			}
		}
		if (miss_flag) {
			miss_num++;
			continue;
		}

		//store sample point 
		point sample_p;
		sample_p.x = x;
		sample_p.y = y;
		sample_points[random_1][random_2] = sample_p;
		flag[random_1][random_2] = true;
		sample_num_1++;
	}

	all_sample_num = sample_num_1;
}

double Sampling::calculatePointsDistance(point p1, point p2) {
	double x_ = p1.x - p2.x;
	double y_ = p1.y - p2.y;
	double dis = sqrt(pow(x_, 2) + pow(y_, 2));
	return dis;
}

double Sampling::scalar_product(point v1, point v2) {
	double res = v1.x * v2.x + v1.y * v2.y;
	return res;
}

double Sampling::cross_product(point v1, point v2) {
	double result = v1.x * v2.y - v2.x * v1.y;
	return result;
}

double Sampling::calculateLeastDistance(point center, line seg, point& least_p) {
	point p1 = seg.p1;
	point p2 = seg.p2;
	point p1_c = center - p1;
	point p1_p2 = p2 - p1;
	double dot_product = scalar_product(p1_c, p1_p2);
	double distance;
	if (dot_product < 0) {
		distance = calculatePointsDistance(center, p1);
		least_p = p1;
	}
	else if (dot_product > pow(calculatePointsDistance(p1, p2), 2)) {
		distance = calculatePointsDistance(center, p2);
		least_p = p2;
	}
	else {
		double temp = dot_product / pow(calculatePointsDistance(p1, p2), 2);
		point v = temp * p1_p2;
		point project_p = p1 + v;
		distance = calculatePointsDistance(project_p, center);
		least_p = project_p;
	}
	return distance;
}

bool Sampling::computeIntersectSegAndCircle(line seg, point center, double radius, std::vector<point>& intersect_points) {
	point p1 = seg.p1;
	point p2 = seg.p2;
	double dis1 = calculatePointsDistance(p1, center);
	double dis2 = calculatePointsDistance(p2, center);
	bool p1_near = dis1 > radius ? false : true;
	bool p2_near = dis2 > radius ? false : true;
	//one in circle one out circle
	if (p1_near ^ p2_near) {
		point intersect;
		if (p1.x == p2.x) {
			intersect.x = p1.x;     //y=?
			double bias = sqrt(pow(radius, 2) - pow(intersect.x - center.x, 2));
			double y1 = center.y + bias;
			double y2 = center.y - bias;
			double min_ = std::min(p1.y, p2.y);
			double max_ = std::max(p1.y, p2.y);
			if (y1 > min_ && y1 < max_) {
				intersect.y = y1;
			}
			if (y2 > min_ && y2 < max_) {
				intersect.y = y2;
			}
		}
		else if (p1.y == p2.y) {
			intersect.y = p1.y;   //x=?
			double bias = sqrt(pow(radius, 2) - pow(intersect.y - center.y, 2));
			double x1 = center.x + bias;
			double x2 = center.x - bias;
			double min_ = std::min(p1.x, p2.x);
			double max_ = std::max(p1.x, p2.x);
			if (x1 > min_ && x1 < max_) {
				intersect.x = x1;
			}
			if (x2 > min_ && x2 < max_) {
				intersect.x = x2;
			}
		}
		else {
			//compute unit vector of line p1,p2
			point p1_p2 = p2 - p1;
			double dis = calculatePointsDistance(p1, p2);
			point unit_v;
			unit_v.x = p1_p2.x / dis;
			unit_v.y = p1_p2.y / dis;

			//compute project point of center on line p1,p2
			point p1_c = center - p1;
			double dot_product = scalar_product(p1_c, p1_p2);
			double tmp = dot_product / dis;
			point project_p = tmp * unit_v;
			project_p = p1 + project_p;
			
			//compute two intersect point on line p1,p2
			double project_dis = calculatePointsDistance(project_p, center);
			double base = sqrt(pow(r, 2) - pow(project_dis, 2));
			point intersect_1 = base * unit_v;
			intersect_1 = project_p - intersect_1;
			point intersect_2 = base * unit_v;
			intersect_2 = project_p + intersect_2;

			//choose one intersect point on seg p1,p2
			point p1_intersect_1 = intersect_1 - p1;
			point p2_intersect_1 = intersect_1 - p2;
			point p1_intersect_2 = intersect_2 - p1;
			point p2_intersect_2 = intersect_2 - p2;
			bool on_seg_1 = scalar_product(p1_intersect_1, p2_intersect_1) < 0 ? true : false;
			bool on_seg_2 = scalar_product(p1_intersect_2, p2_intersect_1) < 0 ? true : false;
			if (on_seg_1) {
				intersect = intersect_1;
			}
			else {
				intersect = intersect_2;
			}
		}
		intersect_points.push_back(intersect);
		return true;
	}
	//both in circle
	if (p1_near && p2_near)
		return 0;
	//both out circle
	if (!p1_near && !p2_near) {
		point least_p;
		double distance = calculateLeastDistance(center, seg, least_p);
		if (distance > radius) {
			return false;
		}
		if (distance == radius) {
			//intersect_points.push_back(least_p);
			//return true;
			return false;
		}
		if (distance < radius) {
			point intersect_1, intersect_2;
			if (p1.x == p2.x) {
				intersect_1.x = intersect_2.x = p1.x;   //y=?
				double bias = sqrt(pow(radius, 2) - pow(p1.x - center.x, 2));
				intersect_1.y = center.y + bias;
				intersect_2.y = center.y - bias;
			}
			else if (p1.y == p2.y) {
				intersect_1.y = intersect_2.y = p1.y;   //x=?
				double bias = sqrt(pow(radius, 2) - pow(p1.y - center.y, 2));
				intersect_1.x = center.x + bias;
				intersect_2.x = center.x - bias;
			}
			else {
				//compute unit vector of line p1,p2
				point p1_p2 = p2 - p1;
				double dis = calculatePointsDistance(p1, p2);
				point unit_v;
				unit_v.x = p1_p2.x / dis;
				unit_v.y = p1_p2.y / dis;

				//compute project point of center on line p1,p2
				point p1_c = center - p1;
				double dot_product = scalar_product(p1_c, p1_p2);
				double tmp = dot_product / dis;
				point project_p = tmp * unit_v;
				project_p = p1 + project_p;

				//compute two intersect point on line p1,p2
				double project_dis = calculatePointsDistance(project_p, center);
				double base = sqrt(pow(r, 2) - pow(project_dis, 2));
				intersect_1 = base * unit_v;
				intersect_1 = project_p - intersect_1;
				intersect_2 = base * unit_v;
				intersect_2 = project_p + intersect_2;
			}
			intersect_points.push_back(intersect_1);
			intersect_points.push_back(intersect_2);
			return true;
		}
	}
}

void Sampling::doubleCopyPointVector(std::vector<point>source_v, std::vector<point>& target_v) {
	int source_v_size = source_v.size();
	for (int i = 0; i < source_v_size; i++) {
		point p1 = source_v[i];
		point p2 = source_v[(i + 1) % source_v_size];
		target_v.push_back(p1);
		target_v.push_back(p2);
	}
}

//void Sampling::halfCopyPointVector(std::vector<point>source_v, std::vector<point>& target_v) {
//	target_v.clear();
//	int source_v_size = source_v.size();
//	for (int i = 0; i < source_v_size; i++) {
//		point p1 = source_v[i];
//		point p2 = source_v[(i + 1) % source_v_size];
//		if (p1 != p2) {
//			target_v.push_back(p1);
//		}
//	}
//}

void Sampling::halfCopyPointVector(std::vector<std::pair<point, int>>source_v, std::vector<std::pair<point, int>>& target_v) {
	target_v.clear();
	int source_v_size = source_v.size();
	for (int i = 0; i < source_v_size; i++) {
		point p1 = source_v[i].first;
		point p2 = source_v[(i + 1) % source_v_size].first;
		if (p1 != p2) {
			std::pair<point, int> pair1(p1, source_v[i].second);
			target_v.push_back(pair1);
		}
	}
}

bool Sampling::calculateCirclesIntersect(point center1, point center2, double radius, std::vector<point>& intersect_points) {
	double dis = calculatePointsDistance(center1, center2);
	if (dis > 2 * radius) {
		return false;
	}
	else if (dis < 2 * radius) {
		point mid_c1_c2;
		mid_c1_c2.x = (center1.x + center2.x) / 2;
		mid_c1_c2.y = (center1.y + center2.y) / 2;
		point a;
		a.x = (center2.x - center1.x) / dis;
		a.y = (center2.y - center1.y) / dis;
		point b;
		b.x = (center2.y - center1.y) / dis;
		b.y = (center1.x - center2.x) / dis;
		double b_ = sqrt(pow(radius, 2) - pow(dis / 2, 2));
		point intersect_1 = b_ * b;
		intersect_1 = mid_c1_c2 + intersect_1;
		point intersect_2 = b_ * b;
		intersect_2 = mid_c1_c2 - intersect_2;
		intersect_points.push_back(intersect_1);
		intersect_points.push_back(intersect_2);
		return true;
	}
	else {
		point intersect;
		intersect.x = (center1.x + center2.x) / 2;
		intersect.y = (center1.y + center2.y) / 2;
		intersect_points.push_back(intersect);
		return true;
	}
}

bool Sampling::isPointInPolygon(point query_p, std::vector<std::pair<point, int>> polygon) {
	int vertex_num = polygon.size();
	bool cross_product_dir;
	for (int i = 0; i < vertex_num; i++) {
		point p1 = polygon[i].first;
		point p2 = polygon[(i + 1) % vertex_num].first;
		point p1_p2 = p2 - p1;
		point p1_q = query_p - p1;
		double tmp = cross_product(p1_p2, p1_q);
		if (i == 0) {
			cross_product_dir = tmp > 0 ? true : false;
		}
		else {
			bool tmp_dir = tmp > 0 ? true : false;
			if (tmp_dir ^ cross_product_dir)
				return false;
		}
	}
	return true;
}

void Sampling::initializePolygonVoid() {
	//traverse every cell
	for (int i = 0; i < grid_num; i++) {
		std::vector<cell*> row_cells;
		for (int j = 0; j < grid_num; j++) {
			cell* cell_ = new cell();
			cell_->idx_1 = i;
			cell_->idx_2 = j;
			row_cells.push_back(cell_);
			//invalid cell
			if (flag[i][j]) {
				cell_->sample = true;
				continue;
			}
			//valid cell,initialize void_polygon,polygons
			neighbor* neighbor_ = new neighbor();
			cell_->neighbor_ = neighbor_;
			int circle_num = 0;
			point left_down;
			left_down.x = grid_size * j;
			left_down.y = grid_size * i;
			point right_down;
			right_down.x = left_down.x + grid_size;
			right_down.y = left_down.y;
			point right_up;
			right_up.x = right_down.x;
			right_up.y = right_down.y + grid_size;
			point left_up;
			left_up.x = left_down.x;
			left_up.y = right_up.y;
			std::pair<point, int> pair1(left_down, -1);
			std::pair<point, int> pair2(right_down, -1);
			std::pair<point, int> pair3(right_up, -1);
			std::pair<point, int> pair4(left_up, -1);
			cell_->void_polygon.push_back(pair1);
			cell_->void_polygon.push_back(pair2);
			cell_->void_polygon.push_back(pair3);
			cell_->void_polygon.push_back(pair4);

			cell_->polygons.push_back(cell_->void_polygon);

			//initialize neighbor_circles
			for (int k = -2; k < 3; k++) {
				for (int l = -2; l < 3; l++) {
					if (abs(k) == 2 && abs(l) == 2)
						continue;
					int idx1 = i + k;
					int idx2 = j + l;
					if (idx1<0 || idx1>grid_num - 1 || idx2<0 || idx2>grid_num - 1)
						continue;
					if (flag[idx1][idx2]) {
						circle_num++;
						circle circle_;
						circle_.center = sample_points[idx1][idx2];
						circle_.radius = r;
						neighbor_->neighbor_circles.push_back(circle_);
						neighbor_->totally_uncover.push_back(false);
					}
				}
			}
			neighbor_->circle_num = circle_num;
		}
		all_cells.push_back(row_cells);
	}
}

void Sampling::updatePolygonVoid(cell* cell_, int circle_idx, int void_idx, std::vector<std::pair<point, int>>& updated_void_polygon) {
	bool totally_covered = true;
	bool totally_uncovered = true;
	circle circle_ = cell_->neighbor_->neighbor_circles[circle_idx];
	point center = circle_.center;
	//int void_len_num = cell_->void_polygon.size();    //the updated void_polygon
	int void_len_num = cell_->polygons[void_idx].size();
	std::vector<std::pair<point, int>> void_polygon = cell_->polygons[void_idx];
	std::vector<std::pair<point, int>> void_polygon_update;

	//for every len of the void_polygon
	for (int m = 0; m < void_len_num; m++) {
		line void_line;
		void_line.p1 = void_polygon[m].first;
		void_line.p2 = void_polygon[(m + 1) % void_len_num].first;
		int len_p1 = void_polygon[m].second;
		int len_p2 = void_polygon[(m + 1) % void_len_num].second;
		std::vector<point> intersect_points;
		bool has_intersect = computeIntersectSegAndCircle(void_line, center, r, intersect_points);
		int intersect_num = intersect_points.size();
		if (intersect_num == 1) {
			double dis1 = calculatePointsDistance(void_line.p1, center);
			double dis2 = calculatePointsDistance(void_line.p2, center);
			if (dis1 < r) {
				std::pair<point, int> pair1(intersect_points[0], circle_idx);
				void_polygon_update.push_back(pair1);
				std::pair<point, int> pair2(void_line.p2, len_p2);
				void_polygon_update.push_back(pair2);
				totally_covered = false;
				totally_uncovered = false;
			}
			if (dis2 < r) {
				std::pair<point, int> pair1(void_line.p1, len_p1);
				void_polygon_update.push_back(pair1);
				std::pair<point, int> pair2(intersect_points[0], circle_idx);
				void_polygon_update.push_back(pair2);
				totally_covered = false;
				totally_uncovered = false;
			}
		}
		if (intersect_num == 0) {
			double dis1 = calculatePointsDistance(void_line.p1, center);
			if (dis1 > r) {
				std::pair<point, int> pair1(void_line.p1, len_p1);
				void_polygon_update.push_back(pair1);
				std::pair<point, int> pair2(void_line.p2, len_p2);
				void_polygon_update.push_back(pair2);
				totally_covered = false;             //the line totally outside the circle
			}
		}
		if (intersect_num == 2) {
			std::pair<point, int> pair1(void_line.p1, len_p1);
			void_polygon_update.push_back(pair1);
			std::pair<point, int> pair2(intersect_points[0], circle_idx);
			void_polygon_update.push_back(pair2);
			std::pair<point, int> pair3(intersect_points[1], circle_idx);
			void_polygon_update.push_back(pair3);
			std::pair<point, int> pair4(void_line.p2, len_p2);
			void_polygon_update.push_back(pair4);
			totally_covered = false;
			totally_uncovered = false;
		}
	}
	//the circle totally uncover the void
	//if (totally_uncovered && !totally_covered) {
	//	cell_->neighbor_->totally_uncover[circle_idx] = true;//
	//	//continue;
	//	return;
	//}
	
	//all the len of void_polygon have no intersect with the circle
	if (totally_covered) {
		if (totally_covered) {
			//no need to check other neighboring circle
			//flag[i][j] = true;   //no void to sample
			return;
		}
	}
	else {
		//update the circle's impact on void_polygon
		//halfCopyPointVector(void_polygon_update, cell_->void_polygon);
		//halfCopyPointVector(void_polygon_update, cell_->polygons[void_idx]);
		halfCopyPointVector(void_polygon_update, updated_void_polygon);
	}
}

void Sampling::finetunePolygonVoid(cell* cell_, int circle_idx, std::vector<std::pair<point, int>>& updated_void_polygon) {
	circle circle_ = cell_->neighbor_->neighbor_circles[circle_idx];
	point center = circle_.center;
	for (int t = circle_idx - 1; t >= 0; t--) {
		//compute two circle intersect point(0 or 1 or 2) in polygon
		circle circle_pre = cell_->neighbor_->neighbor_circles[t];
		point circle_pre_c = circle_pre.center;
		std::vector<point> intersect_points;
		std::vector<point> intersectInPolygon;
		if (calculateCirclesIntersect(center, circle_pre_c, r, intersect_points)) {
			for (int i = 0; i < intersect_points.size(); i++) {
				point intersect = intersect_points[i];
				if (isPointInPolygon(intersect, updated_void_polygon))
					intersectInPolygon.push_back(intersect);
			}
			//if the point_inside ==0,no impact on void_polygon
			//if the point_inside ==2,find the chord
			int intersect_num = intersectInPolygon.size();
			if (intersect_num == 2) {
				int idx = 0;
				for (std::vector<std::pair<point, int>>::iterator it = updated_void_polygon.begin();
					it != updated_void_polygon.end(); it++) {
					int belong1 = updated_void_polygon[idx].second;
					int belong2 = updated_void_polygon[(idx + 1) % updated_void_polygon.size()].second;
					if (belong1 == belong2) {
						point p1 = updated_void_polygon[idx].first;
						double dis1 = calculatePointsDistance(p1, intersectInPolygon[0]);
						double dis2 = calculatePointsDistance(p1, intersectInPolygon[1]);
						std::pair<point, int> pair1(intersectInPolygon[dis1 < dis2 ? 0 : 1], 400);
						it = updated_void_polygon.insert(it + 1, pair1);
						idx++;
					}
					idx++;
				}
			}
			//if the point ==1
			if ((intersect_num == 1) && (intersect_points.size() == 1)) {

			}
			if ((intersect_num == 1) && (intersect_points.size() > 1)) {
				//find the chord of circle2
				int idx = 0;
				for (std::vector<std::pair<point, int>>::iterator it = updated_void_polygon.begin();
					it != updated_void_polygon.end(); it++) {
					int belong = updated_void_polygon[idx].second;
					point p = updated_void_polygon[idx].first;
					point center_ = cell_->neighbor_->neighbor_circles[t].center;//
					double dis = calculatePointsDistance(p, center_);
					if (belong == circle_idx && dis <= r) {
						std::pair<point, int> pair1(intersectInPolygon[0], 200);
						updated_void_polygon[idx] = pair1;
						break;
					}
					idx++;
				}
			}
		}
	}
}

void Sampling::addPolygons(std::vector<std::vector<std::pair<point, int>>> source, std::vector<std::vector<std::pair<point, int>>>& target) {
	int source_size = source.size();
	for (int i = 0; i < source_size; i++) {
		target.push_back(source[i]);
	}
}

void Sampling::computePolygonVoid() {
	//initialize cell_->polygons
	initializePolygonVoid();

	//compute void for valid cell
	for (int i = 0; i < grid_num; i++) {
		for (int j = 0; j < grid_num; j++) {
			if (flag[i][j])
				continue;

			cell* cell_ = all_cells[i][j];
			int circle_num = cell_->neighbor_->circle_num;
			int void_num = cell_->polygons.size();
			//circle_num = std::min(3, circle_num);
			
			for (int k = 0; k < circle_num; k++) {
				std::vector<std::vector<std::pair<point, int>>> polygons_update;
				for(int e = 0; e < void_num; e++) {
					std::vector<std::pair<point, int>> updated_void_polygon;
					updatePolygonVoid(cell_, k, e, updated_void_polygon);
					//if the polygon is totally covered
					if (updated_void_polygon.size() == 0) {
						//no need to fine tune and no add to polygons_update
						continue;
					}

					//fine tune vertexes of void polygon
					circle circle_ = cell_->neighbor_->neighbor_circles[k];
					point center = circle_.center;
					for (int t = k - 1; t >= 0; t--) {
						//if (!cell_->neighbor_->totally_uncover[t] && !cell_->neighbor_->totally_uncover[k])
						if(true)
						{
							//compute two circle intersect point(0 or 1 or 2) in polygon
							circle circle_pre = cell_->neighbor_->neighbor_circles[t];
							point circle_pre_c = circle_pre.center;
							std::vector<point> intersect_points;
							std::vector<point> intersectInPolygon;
							if (calculateCirclesIntersect(center, circle_pre_c, r, intersect_points)) {
								for (int i = 0; i < intersect_points.size(); i++) {
									point intersect = intersect_points[i];
									if (isPointInPolygon(intersect, updated_void_polygon))
										intersectInPolygon.push_back(intersect);
								}
								//if the point_inside ==0,no impact on void_polygon
								//if the point_inside ==2,find the chord
								int intersect_num = intersectInPolygon.size();
								if (intersect_num == 2) {
									int idx = 0;
									for (std::vector<std::pair<point, int>>::iterator it = updated_void_polygon.begin();
										it != updated_void_polygon.end(); it++) {
										int belong1 = updated_void_polygon[idx].second;
										int belong2 = updated_void_polygon[(idx + 1) % updated_void_polygon.size()].second;
										if (belong1 == belong2) {
											point p1 = updated_void_polygon[idx].first;
											double dis1 = calculatePointsDistance(p1, intersectInPolygon[0]);
											double dis2 = calculatePointsDistance(p1, intersectInPolygon[1]);
											std::pair<point, int> pair1(intersectInPolygon[dis1 < dis2 ? 0 : 1], 400);
											//std::vector<std::pair<point, int>>::iterator it_ = it + 1;
											it = updated_void_polygon.insert(it + 1, pair1);
											//it++;
											idx++;
										}
										idx++;
									}
								}
								//if the point ==1
								if ((intersect_num == 1) && (intersect_points.size() == 1)) {

								}
								if ((intersect_num == 1) && (intersect_points.size() > 1)) {
									//find the chord of circle2
									int idx = 0;
									for (std::vector<std::pair<point, int>>::iterator it = updated_void_polygon.begin();
										it != updated_void_polygon.end(); it++) {
										int belong = updated_void_polygon[idx].second;
										point p = updated_void_polygon[idx].first;
										point center_ = cell_->neighbor_->neighbor_circles[t].center;//
										double dis = calculatePointsDistance(p, center_);
										if (belong == k && dis <= r) {
											std::pair<point, int> pair1(intersectInPolygon[0], 200);
											updated_void_polygon[idx] = pair1;
											break;
										}
										idx++;
									}
								}
							}
						}
					}

					//reorganize the polygon, change the updated_void_polygon
					std::vector<std::vector<std::pair<point, int>>> updated_void_polygons;
					storeVoidPolygons(updated_void_polygon, updated_void_polygons);
					//then add the polygon to polygons_update
					addPolygons(updated_void_polygons, polygons_update);
				}
				//if polygons are totally covered
				if (polygons_update.size() == 0) {
					flag[i][j] = true;
					break;
				}
				cell_->polygons = polygons_update;
			}

			/*
			for (int k = 0; k < circle_num; k++) {
				bool totally_covered = true;
				bool totally_uncovered = true;
				updatePolygonVoid(cell_, k, totally_covered, totally_uncovered);
				if (totally_covered) {
					flag[i][j] = true;
					break;
				}
				circle circle_ = cell_->neighbor_->neighbor_circles[k];
				point center = circle_.center;
				for (int t = k - 1; t > 0; t--) {
					if (!cell_->neighbor_->totally_uncover[t] && !cell_->neighbor_->totally_uncover[k]) {
						//compute two circle intersect point(0 or 1 or 2) in polygon
						circle circle_pre = cell_->neighbor_->neighbor_circles[k - 1];
						point circle_pre_c = circle_pre.center;
						std::vector<point> intersect_points;
						std::vector<point> intersectInPolygon;
						if (calculateCirclesIntersect(center, circle_pre_c, r, intersect_points)) {
							for (int i = 0; i < intersect_points.size(); i++) {
								point intersect = intersect_points[i];
								if (isPointInPolygon(intersect, cell_->void_polygon))
									intersectInPolygon.push_back(intersect);
							}
							//if the point_inside ==0,no impact on void_polygon
							//if the point_inside ==2,find the chord
							int intersect_num = intersectInPolygon.size();
							if (intersect_num == 2) {
								int idx = 0;
								for (std::vector<std::pair<point, int>>::iterator it = cell_->void_polygon.begin();
									it != cell_->void_polygon.end(); it++) {
									int belong1 = cell_->void_polygon[idx].second;
									int belong2 = cell_->void_polygon[(idx + 1) % cell_->void_polygon.size()].second;
									if (belong1 == belong2) {
										point p1 = cell_->void_polygon[idx].first;
										double dis1 = calculatePointsDistance(p1, intersectInPolygon[0]);
										double dis2 = calculatePointsDistance(p1, intersectInPolygon[1]);
										std::pair<point, int> pair1(intersectInPolygon[dis1 < dis2 ? 0 : 1], 400);
										//std::vector<std::pair<point, int>>::iterator it_ = it + 1;
										it = cell_->void_polygon.insert(it + 1, pair1);
										//it++;
										idx++;
									}
									idx++;
								}
							}
							//if the point ==1
							if ((intersect_num == 1) && (intersect_points.size() == 1)) {

							}
							if ((intersect_num == 1) && (intersect_points.size() > 1)) {
								//find the chord of circle2
								int idx = 0;
								for (std::vector<std::pair<point, int>>::iterator it = cell_->void_polygon.begin();
									it != cell_->void_polygon.end(); it++) {
									int belong = cell_->void_polygon[idx].second;
									point p = cell_->void_polygon[idx].first;
									point center = cell_->neighbor_->neighbor_circles[t].center;//
									double dis = calculatePointsDistance(p, center);
									if (belong == k && dis < r) {
										std::pair<point, int> pair1(intersectInPolygon[0], 200);
										cell_->void_polygon[idx] = pair1;
										break;
									}
								}
							}

						}
					}
				}
			}
			*/
		}
	}
}

void Sampling::storeVoidPolygons(std::vector<std::pair<point, int>> void_polygon, std::vector<std::vector<std::pair<point, int>>>& polygons) {
	int polygon_num = 0;
	std::vector<int> idexes;
	for (int id = 0; id < void_polygon.size(); id++) {
		if (void_polygon[id].second == 400) {
			idexes.push_back(id);
		}
	}
	polygon_num = std::max(int(idexes.size()), 1);
	if (polygon_num > 1) {
		for (int m = 0; m < polygon_num; m++) {
			std::vector<std::pair<point, int>> polygon1;
			int tmp = idexes[m];
			tmp = (tmp + 1) % void_polygon.size();
			while (void_polygon[tmp].second != 400) {
				polygon1.push_back(void_polygon[tmp]);
				tmp = (tmp + 1) % void_polygon.size();
			}
			polygon1.push_back(void_polygon[tmp]);
			polygons.push_back(polygon1);
		}
	}
	else {
		polygons.push_back(void_polygon);
	}
	/*
	for (int i = 0; i < grid_num; i++) {
		for (int j = 0; j < grid_num; j++) {
			if (!flag[i][j]) {
				cell* cell_ = all_cells[i][j];
				int polygon_num = 0;
				std::vector<int> idexes;
				std::vector<std::vector<std::pair<point, int>>> polygons;
				std::vector<std::pair<point, int>> void_polygon_ = cell_->void_polygon;
				for (int id = 0; id < void_polygon_.size(); id++) {
					if (void_polygon_[id].second == 400) {
						polygon_num++;
						idexes.push_back(id);
					}
				}
				if (polygon_num > 1) {
					for (int m = 0; m < polygon_num; m++) {
						std::vector<std::pair<point, int>> polygon1;
						int tmp = idexes[m];
						tmp = (tmp + 1) % void_polygon_.size();
						while (void_polygon_[tmp].second != 400) {
							polygon1.push_back(void_polygon_[tmp]);
							tmp = (tmp + 1) % void_polygon_.size();
						}
						polygon1.push_back(void_polygon_[tmp]);
					}
				}
				else {
					polygons.push_back(void_polygon_);
				}
				cell_->polygons = polygons;
			}
		}
	}
	*/
}

double Sampling::computePolygonArea(std::vector<std::pair<point, int>> polygon) {
	double area = 0;
	int v_num = polygon.size();
	double x_min = DBL_MAX;
	int x_min_idx;
	for (int i = 0; i < v_num; i++) {
		double x = polygon[i].first.x;
		if (x < x_min) {
			x_min_idx = i;
			x_min = x;
		}
		else if (x == x_min) {
			double y = polygon[i].first.y;
			double y_ = polygon[x_min_idx].first.y;
			if (y < y_) {
				x_min_idx = i;
			}
		}
	}
	int source_idx = x_min_idx;
	int target_idx = source_idx > 0 ? (source_idx - 1) : (source_idx - 1 + v_num);
	for (int i = 0; i < v_num; i++) {
		point source = polygon[source_idx].first;
		point target = polygon[target_idx].first;
		double area_ = 0.5 * (source.y + target.y) * (target.x - source.x);
		area += area_;
		source_idx = target_idx;
		target_idx = source_idx > 0 ? (source_idx - 1) : (source_idx - 1 + v_num);
	}
	return area;
}

int Sampling::randomIndex(std::vector<double> cumsum) {
	std::default_random_engine e(time(0));
	std::uniform_real_distribution<double> u(0, 1);
	double random = u(e);

	//binary search
	int random_idx;
	int low = 0;
	int high = cumsum.size() - 1;
	int mid = 0;
	while (low <= high) {
		mid = (low + high) / 2;
		if (cumsum[mid] < random)
			low = mid + 1;
		else if (cumsum[mid] > random)
			high = mid - 1;
		else {
			random_idx = mid;
			break;
		}
	}
	random_idx = mid;
	return random_idx;
}

void Sampling::randomPointInPolygon(std::vector<std::vector<std::pair<point, int>>> polygons, double polygons_area, point& sample_p) {
	std::vector<std::vector<std::pair<point, int>>> triangles;
	for (int p_idx = 0; p_idx < polygons.size(); p_idx++) {
		std::vector<std::pair<point, int>> polygon = polygons[p_idx];
		if (polygon.size() == 3) {
			triangles.push_back(polygon);
		}
		else {
			point sum;
			sum.x = 0;
			sum.y = 0;
			for (int i = 0; i < polygon.size(); i++) {
				sum.x += polygon[i].first.x;
				sum.y += polygon[i].first.y;
			}
			sum.x /= polygon.size();
			sum.y /= polygon.size();
			for (int i = 0; i < polygon.size(); i++) {
				std::vector<std::pair<point, int>> triangle;
				triangle.push_back(polygon[i]);
				triangle.push_back(polygon[(i + 1) % polygon.size()]);
				std::pair<point, int> pair1(sum, 800);
				triangle.push_back(pair1);
				triangles.push_back(triangle);
			}
		}
	}
	//compute cumulative sum
	std::vector<double> AreaSum;
	for (int i = 0; i < triangles.size(); i++) {
		if (i == 0) {
			AreaSum.push_back(computePolygonArea(triangles[i]) / polygons_area);
		}
		else {
			double cumulative_sum = AreaSum[i - 1] + computePolygonArea(triangles[i]) / polygons_area;
			AreaSum.push_back(cumulative_sum);
		}
	}
	int random_idx = randomIndex(AreaSum);

	//sample in triangle
	point p1 = triangles[random_idx][0].first;
	point p2 = triangles[random_idx][1].first;
	point p3 = triangles[random_idx][2].first;
	point p1_p2 = p2 - p1;
	point p1_p3 = p3 - p1;

	srand((unsigned)time(NULL));
	double a1 = rand() / double(RAND_MAX);
	double a2 = rand() / double(RAND_MAX);
	if ((a1 + a2) > 1) {
		a1 = 1 - a1;
		a2 = 1 - a2;
	}
	p1_p2 = a1 * p1_p2;
	p1_p3 = a2 * p1_p3;
	sample_p = p1 + p1_p2;
	sample_p = sample_p + p1_p3;
}

bool Sampling::CheckneighborCell(int idx_1, int idx_2, point sample_p) {
	for (int k = -2; k < 3; k++) {
		for (int l = -2; l < 3; l++) {
			int idx1 = idx_1 + k;
			int idx2 = idx_2 + l;
			if (idx1 >= 0 && idx1 < grid_num && idx2 >= 0 && idx2 < grid_num) {
				if (flag[idx1][idx2]) {
					point p1 = sample_points[idx1][idx2];
					double dis = calculatePointsDistance(p1, sample_p);
					if (dis < r)
						return false;
				}
			}
		}
	}
	return true;
}

bool Sampling::DartThrowingInVoid() {
	std::vector<double> VoidAreas;
	std::vector<cell*> UnsampleCells;
	double all_voids_area = 0;
	for (int i = 0; i < grid_num; i++) {
		for (int j = 0; j < grid_num; j++) {
			if (!flag[i][j]) {
				cell* cell_ = all_cells[i][j];
				UnsampleCells.push_back(cell_);
				int polygon_num = cell_->polygons.size();
				double voids_area = 0;
				for (int k = 0; k < polygon_num; k++) {
					std::vector<std::pair<point, int>> polygon = cell_->polygons[k];
					double area_ = computePolygonArea(polygon);
					voids_area += area_;
				}
				VoidAreas.push_back(voids_area);
				all_voids_area += voids_area;
			}
		}
	}

	if (UnsampleCells.size() == 0) {
		return false;
	}
	//compute cumulative sum
	std::vector<double> AreaSum;
	for (int i = 0; i < VoidAreas.size(); i++) {
		if (i == 0) {
			AreaSum.push_back(VoidAreas[i] / all_voids_area);
		}
		else {
			double cumulative_sum = AreaSum[i - 1] + VoidAreas[i] / all_voids_area;
			AreaSum.push_back(cumulative_sum);
		}
	}

	int sample_num_2 = 0;
	int miss_times = 0;
	int iteration_num = 3 * UnsampleCells.size();
	std::vector<cell*> new_sampled_cells;
	std::default_random_engine e(time(0));
	for (int i = 0; i < iteration_num; i++) {
		if (miss_times == max_miss_num2)
			break;
		std::uniform_real_distribution<double> u(0, 1);
		double random = u(e);

		//binary search
		int random_idx;
		int low = 0;
		int high = AreaSum.size() - 1;
		int mid = 0;
		while (low <= high) {
			mid = (low + high) / 2;
			if (AreaSum[mid] < random)
				low = mid + 1;
			else if (AreaSum[mid] > random)
				high = mid - 1;
			else {
				random_idx = mid;
				break;
			}
		}
		random_idx = mid;//random choose one cell

		//sample in polygon voids of a cell
		point sample_p;
		randomPointInPolygon(UnsampleCells[random_idx]->polygons, VoidAreas[random_idx], sample_p);
		//check neighboring grids
		int idx_1 = UnsampleCells[random_idx]->idx_1;
		int idx_2 = UnsampleCells[random_idx]->idx_2;
		if (!CheckneighborCell(idx_1, idx_2, sample_p)) {
			miss_times++;
		}
		else {
			//store the sampled point
			sample_num_2++;
			sample_points[idx_1][idx_2] = sample_p;
			flag[idx_1][idx_2] = true;
			new_sampled_cells.push_back(UnsampleCells[random_idx]);
		}
	}

	//update neighboring cells polygon void, add circle information
	int new_sampled_num = new_sampled_cells.size();
	for (int i = 0; i < new_sampled_num; i++) {
		cell* c1 = new_sampled_cells[i];
		int idx_1 = c1->idx_1;
		int idx_2 = c1->idx_2;
		//the new circle
		circle circle1;
		point p1 = sample_points[idx_1][idx_2];
		circle1.center = p1;
		circle1.radius = r;

		for (int k = -2; k < 3; k++) {
			for (int l = -2; l < 3; l++) {
				int idx1 = idx_1 + k;
				int idx2 = idx_2 + l;
				if (idx1 >= 0 && idx1 < grid_num && idx2 >= 0 && idx2 < grid_num) {
					//add the new circle to the neighbor cell's neighbor circles
					cell* neighbor_cell = all_cells[idx1][idx2];
					if (!flag[idx1][idx2]) {
						neighbor_cell->neighbor_->neighbor_circles.push_back(circle1);
						neighbor_cell->neighbor_->circle_num++;
						int new_circle_idx = neighbor_cell->neighbor_->circle_num - 1;

						//update neighbor cell void polygons
						int void_num = neighbor_cell->polygons.size();
						std::vector<std::vector<std::pair<point, int>>> polygons_update;
						for (int e = 0; e < void_num; e++) {
							std::vector<std::pair<point, int>> updated_void_polygon;
							updatePolygonVoid(neighbor_cell, new_circle_idx, e, updated_void_polygon);
							//if the polygon is totally covered
							if (updated_void_polygon.size() == 0)
								continue;
							//fine tune vertexes of void polygon
							finetunePolygonVoid(neighbor_cell, new_circle_idx, updated_void_polygon);

							//reorganize the polygon, change the updated_void_polygon
							std::vector<std::vector<std::pair<point, int>>> updated_void_polygons;
							storeVoidPolygons(updated_void_polygon, updated_void_polygons);

							//then add the polygon to polygons_update
							addPolygons(updated_void_polygons, polygons_update);
						}

						//if polygons are totally covered
						if (polygons_update.size() == 0) {
							flag[idx1][idx2] = true;
							continue;
						}
						neighbor_cell->polygons = polygons_update;
					}
				}
			}
		}
	}

	all_sample_num += sample_num_2;
	return true;
}