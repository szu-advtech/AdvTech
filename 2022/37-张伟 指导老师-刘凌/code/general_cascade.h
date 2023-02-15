#ifndef GENERAL_CASCADE_H
#define GENERAL_CASCADE_H

 
#include "cascade.h"

/// Template class that implements general cascade diffusion
template<class TGraph>
class GeneralCascadeT:
	public CascadeT<TGraph>
{
public:
	int nthreads;

public:
	GeneralCascadeT() : nthreads(1) {}

	virtual void Build(TGraph& gf)
	{
		this->_Build(gf);
	}

	double Run(int num_iter, int size, int set[])
	{
		InitializeConcurrent();
		return _Run(num_iter, size, set);
	}

protected:
	void InitializeConcurrent() 
	{
		if (IsConcurrent())
		{
#ifdef MI_USE_OMP
			// run concurrently
			const double DYNAMIC_RATIO = 0.25;
			omp_set_num_threads(nthreads);
			int dynamicThreads = (int)(nthreads * DYNAMIC_RATIO);
			omp_set_dynamic(dynamicThreads);

			std::cout << "== Turn on omp optimization: == " << std::endl;
			std::cout << "#Max Threads = " << omp_get_max_threads() << "\t#Dynamic Threads = " << omp_get_dynamic() << std::endl;
#else
			std::cout << "== omp is not supported or enabled == " << std::endl;
#endif
		}
	}

	inline bool IsConcurrent()
	{
		return (nthreads > 1);
	}

	double _Run(int num_iter, int size, int set[]) 
	{
		if (this->gf == NULL) {
			throw NullPointerException("Please Build Graph first. (gf==NULL)");
		}
		
		int targetSize = size;
		int resultSize = 0;
		ProbTransfom trans(this->gf->edgeForm);

#ifdef MI_USE_OMP
		if (!IsConcurrent()) {
#endif
			// single thread
			int* list = new int[this->n];
			bool* active = new bool[this->n];
			for (int it = 0; it < num_iter; it++)
			{
				memset(active, 0, sizeof(bool)*(this->n));
				for (int i = 0; i < targetSize; i++)
				{
					list[i] = set[i];
					active[list[i]] = true;
				}
				resultSize += targetSize;

				int h = 0;
				int t = targetSize;

				while (h < t)
				{
					int k = this->gf->GetNeighborCount(list[h]);
					for (int i = 0; i < k; i++)
					{
						auto& e = this->gf->GetEdge(list[h], i);
						if (active[e.v]) continue;

						if ((this->random).RandBernoulli(trans.Prob(e.w1)))
						{
							list[t] = e.v;
							active[e.v] = true;
							t++;
							resultSize++;
						}
					}
					h++;
				}
			}

			SAFE_DELETE_ARRAY(active);
			SAFE_DELETE_ARRAY(list);

#ifdef MI_USE_OMP
		} else {
			// concurrent
			#pragma omp parallel for
			for (int it = 0; it < num_iter; it++) {
				std::vector<int> list(this->n);
				std::vector<bool> active(this->n, false);
				for (int i = 0; i < targetSize; i++)
				{
					list[i] = set[i];
					active[list[i]] = true;
				}
				int curResult = targetSize;
				int h = 0;
				int t = targetSize;
				
				while (h < t)
				{
					int k = this->gf->GetNeighborCount(list[h]);
					for (int i = 0; i < k; i++)
					{
						auto& e = this->gf->GetEdge(list[h], i);
						if (active[e.v]) continue;

						if ((this->random).RandBernoulli(trans.Prob(e.w1)))
						{
							list[t] = e.v;
							active[e.v] = true;
							t++;
							curResult++;
						}
					}
					h++;
				}

				#pragma omp atomic
				resultSize += curResult;
			}
		}
#endif

		return (double)resultSize / (double)num_iter;
	}
};


typedef GeneralCascadeT<Graph> GeneralCascade;


#endif
