#ifndef VINA_INDIVIDUAL_H
#define VINA_INDIVIDUAL_H

#include "model.h"
#include "conf.h"
#include "incrementable.h"
#include <algorithm>
#include "random.h"
#include "energy.h"


bool MyCompare(const output_type& x, const output_type& y);

struct ADE {
	sz num_saved_mins;//最后out的容量
	int num_steps;//迭代次数
	fl min_rmsd;//评估最后的结果
	ADE() : num_steps(3000),num_saved_mins(20), min_rmsd(1.0){ }
	void operator()(model& m, output_container& out, const precalculate& p, const igrid& ig, const precalculate& p_widened, const igrid& ig_widened, const vec& corner1, const vec& corner2, incrementable* increment_me, rng& generator) const;
};

struct ADE_aux {
	int po_size;//种群规模
	int dim;// 维度
	int best_index; //记录当前种群变异最优的个体
	fl c;  //相关系数，0.2
	fl S_F;  //保存当前种群变异更优个体F的 lehmer均值
	fl S_CR;  //保存当前种群变异更优个体CR的 算数平均数
	std::vector<output_type> nowpopulation;//初始种群
	std::vector<output_type> mutatepopulation;//变异交叉后的种群
	ADE_aux() :  po_size(30), dim(7){}
	ADE_aux(const model& m) : po_size(30){
		dim = m.ligand_degrees_of_freedom(0) + 7;
		nowpopulation.resize(po_size);
		mutatepopulation.resize(po_size);
		S_F = 0.5;
		S_CR = 0.5;
		c = 0.2;
	}
	void mutate_cross(model& m, energy_cal& energy, rng& generator, const vec& corner1, const vec& corner2, const int& num_steps);
	void select(model& m, energy_cal& energy, rng& generator);

};








#endif
