#include <cmath>
#include <fstream>
#include "lshade.h"
#include "coords.h"
#include "mutate.h"
#include"tee.h"
#include"adam_cal.h"
#include <sstream>


bool MyCompare(const output_type& x, const output_type& y) {
	return x.e < y.e;
}

void my_normalize_angle(fl& x) {
	while (x < -pi)
		x += 2 * pi;
	while(x > pi)
		x -= 2 * pi;
}

void ADE_aux::mutate_cross(model& m, energy_cal& energy, rng& generator, const vec& corner1, const vec& corner2, const int& num_steps) {
	conf_size s = m.get_size();
	change g(s);
	fl best_e = max_fl;
	best_index = -1;
	std::cout << po_size << std::endl;
	for (int i = 0; i < po_size; i++) {
		nowpopulation[i].CR = random_normal(S_CR, 0.1, generator);
		while (nowpopulation[i].CR < 0 || nowpopulation[i].CR > 1) {
			nowpopulation[i].CR = random_normal(S_CR, 0.1, generator);
		}
		nowpopulation[i].F = random_cauchy(S_F, 0.1, generator);
		while (nowpopulation[i].F <= 0 || nowpopulation[i].F > 1) {
			nowpopulation[i].F = random_cauchy(S_F, 0.1, generator);
		}
		int pbest = random_int(0, 4, generator);
		int index1, index2;  
		index1 = random_int(0, po_size - 1, generator);
		while (i == index1) {  
			index1 = random_int(0, po_size - 1, generator);
		}
		index2 = random_int(0, po_size - 1, generator);
		while (index1 == index2 || index2 == i) {  
			index2 = random_int(0, po_size - 1, generator);
		}
		int dim_j = random_int(0, dim - 1, generator);
		for (int j = 0; j < dim; j++) {
			double p_cr = random_fl(0, 1.0, generator);
			if (p_cr < nowpopulation[i].CR || j == dim_j) {
				if (j < 3) {
					mutatepopulation[i].c.ligands[0].rigid.position[j] = nowpopulation[pbest](j) + nowpopulation[i].F * (nowpopulation[index1](j) - nowpopulation[index2](j));
					if (mutatepopulation[i].c.ligands[0].rigid.position[j] < corner1[j]) {
						mutatepopulation[i].c.ligands[0].rigid.position[j] = corner1[j];
					}
					else if (mutatepopulation[i].c.ligands[0].rigid.position[j] > corner2[j]) {
						mutatepopulation[i].c.ligands[0].rigid.position[j] = corner2[j];
					}
				}
				else if (j == 3) {
					mutatepopulation[i].angle = nowpopulation[pbest].angle + nowpopulation[i].F * (nowpopulation[index1].angle - nowpopulation[index2].angle);
					if (mutatepopulation[i].angle != mutatepopulation[i].angle) {
						std::cout << nowpopulation[pbest].angle << " " << nowpopulation[index1].angle << " " << nowpopulation[index2].angle << std::endl;
						mutatepopulation[i].angle = random_fl(-pi, pi, generator);
					}
					//normalize_angle(mutatepopulation[i].angle);
				}
				else if (j > 3 && j <= 6) {
					mutatepopulation[i].axis[j - 4] = nowpopulation[pbest].axis[j - 4] + nowpopulation[i].F * (nowpopulation[index1].axis[j - 4] - nowpopulation[index2].axis[j - 4]);
				}
				else {
					mutatepopulation[i].c.ligands[0].torsions[j - 7] = nowpopulation[pbest](j) + 
						nowpopulation[i].F * (nowpopulation[index1].c.ligands[0].torsions[j - 7] - nowpopulation[index2].c.ligands[0].torsions[j - 7]);
					if (mutatepopulation[i].angle != mutatepopulation[i].angle) {
						std::cout << nowpopulation[pbest](j) << " " << nowpopulation[index1](j) << " " << nowpopulation[index2](j) << std::endl;
						mutatepopulation[i].angle = random_fl(-pi, pi, generator);
					}
					normalize_angle(mutatepopulation[i].c.ligands[0].torsions[j - 7]);
				}
			}
			else {
				if (j < 3) {
					mutatepopulation[i].c.ligands[0].rigid.position[j] = nowpopulation[i](j);
				}
				else if (j == 3) {
					mutatepopulation[i].angle = nowpopulation[i].angle;
					normalize_angle(mutatepopulation[i].angle);
				}
				else if (j > 3 && j <= 6) {
					mutatepopulation[i].axis[j - 4] = nowpopulation[i].axis[j - 4];

				}
				else {
					mutatepopulation[i].c.ligands[0].torsions[j - 7] = nowpopulation[i].c.ligands[0].torsions[j - 7];
					normalize_angle(mutatepopulation[i].c.ligands[0].torsions[j - 7]);
				}
			}
		}
		axis_normalize(mutatepopulation[i].axis);
		mutatepopulation[i].c.ligands[0].rigid.orientation = angle_to_quaternion(mutatepopulation[i].axis, mutatepopulation[i].angle);
		quaternion_normalize(mutatepopulation[i].c.ligands[0].rigid.orientation);
		mutatepopulation[i].e = energy(mutatepopulation[i].c, g);
		if (mutatepopulation[i].e < best_e) {
			best_e = mutatepopulation[i].e;
			best_index = i;
		}
	}
}	


void ADE_aux::select(model& m, energy_cal& energy, rng& generator) {
	std::vector<double> SCR;
	std::vector<double> SF;
	
	for (int i = 0; i < po_size; i++) {
		if (nowpopulation[i].e <= mutatepopulation[i].e) {
			continue;
		}
		else {
			SCR.push_back(nowpopulation[i].CR);
			SF.push_back(nowpopulation[i].F);
			nowpopulation[i] = mutatepopulation[i];
		}
	}

	if (!SF.empty()) {
		double numerator_F = 0;
		double denominator_F = 0;
		for (int i = 0; i < SF.size(); i++) {
			numerator_F += SF[i] * SF[i];
			denominator_F += SF[i];
		}
		S_F = (1 - c) * S_F + c * numerator_F / denominator_F;
	}
	if (!SCR.empty()) {
		fl sum = 0;
		for (int i = 0; i < SCR.size(); i++) {
			sum += SCR[i];
		}
		S_CR = (1 - c) * S_CR + c * sum / SCR.size();
	}
}

path my_make_path(const std::string& str) {
	return path(str);
}

void my_write_all_output(model& m, const output_container& out, sz how_many, const std::string& output_name, const std::vector<std::string>& remarks) {
	if (out.size() < how_many)
		how_many = out.size();
	VINA_CHECK(how_many <= remarks.size());
	ofile f(my_make_path(output_name));
	VINA_FOR(i, how_many) {
		m.set(out[i].c);
		m.write_model(f, i + 1, remarks[i]); // so that model numbers start with 1
	}
}

void ADE::operator()(model& m, output_container& out, const precalculate& p, const igrid& ig, const precalculate& p_widened, const igrid& ig_widened, const vec& corner1, const vec& corner2, incrementable* increment_me, rng& generator) const
{
	//output_container out_iter;
	//std::string out_iter_name = "1sqa_iter.pdbqt";
	//std::vector<std::string> remarks;
	//int many = 0;

	vec authentic_v(1000, 1000, 1000);
	vec v(10, 10, 10);
	fl best_e = max_fl;
	conf_size s = m.get_size();
	change g(s);
	energy_cal energy(&m, &p, &ig, v);
	ADE_aux aux(m);
	output_type tmp(s, 0);
	int tors = m.ligand_degrees_of_freedom(0);
	for (int i = 0; i < aux.po_size; i++) {
		tmp.c.randomize(corner1, corner2, generator);
		tmp.angle = random_fl(-pi, pi, generator);
		for (int j = 0; j < 3; j++) {
			tmp.axis[j] = random_fl(-1, 1, generator);
		}
		axis_normalize(tmp.axis);
		tmp.c.ligands[0].rigid.orientation = angle_to_quaternion(tmp.axis, tmp.angle);
		tmp.e = energy(tmp.c, g);
		aux.nowpopulation[i] = tmp;
		aux.mutatepopulation[i] = tmp;
	}
	std::sort(aux.nowpopulation.begin(), aux.nowpopulation.end(), MyCompare);
	
	std::ofstream ofs("4abg.txt");
	for (int i = 0; i < num_steps; i++) {
		if (increment_me)
			++(*increment_me);
		aux.mutate_cross(m, energy, generator, corner1, corner2, i);
		ofs << "iterations " << i << ":" << std::endl;
		int po_size_cur = aux.nowpopulation.size();
		for (int j = 0; j < po_size_cur; j++) {  
			
			ofs << "origin:" << std::setw(9) << std::setprecision(5) << aux.nowpopulation[j].e << "\t";
			ofs << "mutate:" << std::setw(9) << std::setprecision(5) << aux.mutatepopulation[j].e << "\t";
			ofs << "cr:" << std::setw(9) << std::setprecision(5) << aux.nowpopulation[j].CR << "\t";
			ofs << "f:" << std::setw(9) << std::setprecision(5) << aux.nowpopulation[j].F << "\t";
			ofs << "angle: " << std::setw(9) << std::setprecision(5) << aux.nowpopulation[j].angle << "\t";
			ofs << "axis: " << std::setw(9) << std::setprecision(5) << aux.nowpopulation[j].axis[0] << " " << aux.nowpopulation[j].axis[1] << " " << aux.nowpopulation[j].axis[2] << "\t";
			ofs << "torsions: ";
			for (int k = 0; k < tors; k++) {
				ofs << std::setw(9) << std::setprecision(5) << aux.nowpopulation[j].axis[0] << " " << aux.nowpopulation[j].c.ligands[0].torsions[k] << " ";
			}
			ofs << std::endl;
		}
		aux.select(m, energy, generator);
		std::sort(aux.nowpopulation.begin(), aux.nowpopulation.end(), MyCompare);

		//my_write_all_output(m, out_iter, many, out_iter_name, remarks);

		if (aux.nowpopulation[0].e < best_e || out.size() < num_saved_mins) {
			m.set(aux.nowpopulation[0].c);
			aux.nowpopulation[0].coords = m.get_heavy_atom_movable_coords();
			add_to_output_container(out, aux.nowpopulation[0], min_rmsd, num_saved_mins); // 20 - max size
			if (aux.nowpopulation[0].e < best_e)
				best_e = aux.nowpopulation[0].e;
		}
	}
	//std::cout << "best:" << aux.nowpopulation[0].e << std::endl;
	VINA_CHECK(!out.empty());
	VINA_CHECK(out.front().e <= out.back().e); // make sure the sorting worked in the correct order

}
