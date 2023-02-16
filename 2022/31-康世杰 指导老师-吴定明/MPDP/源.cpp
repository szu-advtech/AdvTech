#include <list>
#include <vector>
#include <stack>
#include <iostream>
#include <cmath>
#include <cstdint>

#include "optimizer/gpuqo_common.h"

#include "gpuqo.cuh"
#include "gpuqo_timing.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_cost.cuh"
#include "gpuqo_filter.cuh"
#include "gpuqo_cpu_dpsub.cuh"
#include "gpuqo_dpsub.cuh"
#include "gpuqo_binomial.cuh"

using namespace std;

template<typename BitmapsetN>
struct call_stack_el_t {
    unsigned int u;
    BitmapsetN edges;
    bool remain;
};

struct edge_stack_el_t {
    unsigned int u;
    unsigned int v;
};

template<typename BitmapsetN, typename memo_t, bool manage_best>
class DPsubBiCCCPUAlgorithm : public DPsubGenericCPUAlgorithm<BitmapsetN, memo_t, manage_best> {
private:

    BitmapsetN output_block(unsigned int u, unsigned int v,
        stack<edge_stack_el_t>& edge_stack) {
        BitmapsetN block(0);

        edge_stack_el_t el;

        do {
            el = edge_stack.top();
            edge_stack.pop();

            block.set(el.u);
            block.set(el.v);
        } while (el.u != u || el.v != v);

        LOG_DEBUG("output_comp(%d, %d): %u\n", u, v, block.toUint());

        return block;
    }

    void find_blocks(BitmapsetN set, list<BitmapsetN>& blocks) {
        stack<call_stack_el_t<BitmapsetN>> call_stack;
        stack<edge_stack_el_t> edge_stack;
        vector<bool> visited(BitmapsetN::SIZE, false);
        vector<int> depth(BitmapsetN::SIZE, -1);
        vector<unsigned int> parent(BitmapsetN::SIZE, 0);
        vector<int> low(BitmapsetN::SIZE, numeric_limits<int>::max());

        auto info = CPUAlgorithm<BitmapsetN, memo_t>::info;

        LOG_DEBUG("find_blocks(%u)\n", set.toUint());

        call_stack.push((call_stack_el_t<BitmapsetN>) {
            .u = set.lowestPos(),
                .edges = BitmapsetN(0),
                .remain = false
        });
        int curr_depth = 1;
        while (!call_stack.empty()) {
            call_stack_el_t<BitmapsetN> el = call_stack.top();
            call_stack.pop();
            if (el.edges.empty()) {
                visited[el.u] = true;
                depth[el.u] = curr_depth;
                LOG_DEBUG("visited %d at depth %d\n", el.u, curr_depth);
                curr_depth++;
                BitmapsetN edges = info->edge_table[el.u - 1] & set;
                if (!edges.empty()) {
                    call_stack.push((call_stack_el_t<BitmapsetN>) {
                        .u = el.u,
                            .edges = edges,
                            .remain = false
                    });
                }
            }
            else {
                unsigned int v = el.edges.lowestPos();
                LOG_DEBUG("edge %d %d: ", el.u, v);
                if (el.remain) {
                    if (low[v] >= depth[el.u]) {
                        blocks.push_back(output_block(el.u, v, edge_stack));
                    }
                    low[el.u] = min(low[el.u], low[v]);
                    LOG_DEBUG("remain, update low of %d: %d\n", el.u, low[el.u]);
                    el.edges.unset(v);
                    if (!el.edges.empty()) {
                        call_stack.push((call_stack_el_t<BitmapsetN>) {
                            .u = el.u,
                                .edges = el.edges,
                                .remain = false
                        });
                    }
                }
                else {
                    if (!visited[v]) {
                        edge_stack.push((edge_stack_el_t) {
                            .u = el.u,
                                .v = v
                        });
                        parent[v] = el.u;
                        call_stack.push((call_stack_el_t<BitmapsetN>) {
                            .u = el.u,
                                .edges = el.edges,
                                .remain = true
                        });
                        LOG_DEBUG("queue visit to %d\n", v);
                        call_stack.push((call_stack_el_t<BitmapsetN>) {
                            .u = v,
                                .edges = BitmapsetN(0),
                                .remain = false
                        });
                    }
                    else if (parent[el.u] != v && depth[v] < depth[el.u]) {
                        edge_stack.push((edge_stack_el_t) {
                            .u = el.u,
                                .v = v
                        });
                        low[el.u] = min(low[el.u], depth[v]);
                        LOG_DEBUG("update low of %d: %d", el.u, low[el.u]);
                        el.edges.unset(v);
                        if (!el.edges.empty()) {
                            call_stack.push((call_stack_el_t<BitmapsetN>) {
                                .u = el.u,
                                    .edges = el.edges,
                                    .remain = false
                            });
                        }
                        else {
                            LOG_DEBUG(" (end %d)", el.u);
                        }
                        LOG_DEBUG("\n");
                    }
                    else {
                        LOG_DEBUG("skip");
                        el.edges.unset(v);
                        if (!el.edges.empty()) {
                            call_stack.push((call_stack_el_t<BitmapsetN>) {
                                .u = el.u,
                                    .edges = el.edges,
                                    .remain = false
                            });
                        }
                        else {
                            LOG_DEBUG(" (end %d)", el.u);
                        }
                        LOG_DEBUG("\n");
                    }
                }
            }
        }
    }

public:
    virtual JoinRelationCPU<BitmapsetN>* enumerate_subsets(BitmapsetN set) {
        list<BitmapsetN> blocks;
        auto info = CPUAlgorithm<BitmapsetN, memo_t>::info;

        find_blocks(set, blocks);

        JoinRelationCPU<BitmapsetN>* join_rel = NULL;

        for (BitmapsetN block : blocks) {
            BitmapsetN lb_id = block.lowest();
            BitmapsetN rb_id;
            while (lb_id != block) {
                rb_id = block - lb_id;

                LOG_DEBUG("Trying %u and %u\n", lb_id.toUint(), rb_id.toUint());

                BitmapsetN left_id = grow(lb_id.lowest(),
                    set - rb_id,
                    info->edge_table);
                BitmapsetN right_id = grow(rb_id.lowest(),
                    set - lb_id,
                    info->edge_table);

                LOG_DEBUG("Grown: %u and %u\n",
                    left_id.toUint(), right_id.toUint());

#ifdef GPUQO_PRINT_N_JOINS
                CPUAlgorithm<BitmapsetN, memo_t>::n_checks++;
#endif
                if ((left_id | right_id) == set) {
                    auto& memo = *CPUAlgorithm<BitmapsetN, memo_t>::memo;
                    auto left = memo.find(left_id);
                    auto right = memo.find(right_id);

                    Assert(left != memo.end() && right != memo.end());
                    JoinRelationCPU<BitmapsetN>* left_rel = left->second;
                    JoinRelationCPU<BitmapsetN>* right_rel = right->second;
                    int level = set.size();

                    JoinRelationCPU<BitmapsetN>* new_join_rel =
                        (*CPUAlgorithm<BitmapsetN, memo_t>::join)(
                            level, false, *left_rel, *right_rel);

                    if (manage_best) {
                        if (join_rel == NULL
                            || (new_join_rel != NULL
                                && new_join_rel->cost.total < join_rel->cost.total))
                        {
                            if (join_rel != NULL)
                                delete join_rel;

                            join_rel = new_join_rel;
                        }
                        else {
                            if (new_join_rel != NULL)
                                delete new_join_rel;
                        }
                    }
                }

                lb_id = nextSubset(lb_id, block);
            }
        }

        return join_rel;
    }

    virtual bool check_join(int level, JoinRelationCPU<BitmapsetN>& left_rel,
        JoinRelationCPU<BitmapsetN>& right_rel)
    {

        // Sets are already checked
#ifdef USE_ASSERT_CHECKING
        auto& info = CPUAlgorithm<BitmapsetN, memo_t>::info;
#endif
        Assert(is_connected(left_rel.id, info->edge_table));
        Assert(is_connected(right_rel.id, info->edge_table));
        Assert(are_connected_rel(left_rel, right_rel, info));
        Assert(is_disjoint_rel(left_rel, right_rel));

        return true;
    }
};


/* gpuqo_cpu_dpsub_bicc
 *
 *	 Sequential CPU baseline for GPU query optimization using the DP sub
 *   algorithm with BiCC optimization.
 */
template<typename BitmapsetN>
QueryTree<BitmapsetN>*
gpuqo_cpu_dpsub_bicc(GpuqoPlannerInfo<BitmapsetN>* info)
{
    DPsubBiCCCPUAlgorithm<BitmapsetN, hashtable_memo_t<BitmapsetN>, false> alg;
    return gpuqo_cpu_sequential(info, &alg);
}

template QueryTree<Bitmapset32>* gpuqo_cpu_dpsub_bicc<Bitmapset32>(GpuqoPlannerInfo<Bitmapset32>*);
template QueryTree<Bitmapset64>* gpuqo_cpu_dpsub_bicc<Bitmapset64>(GpuqoPlannerInfo<Bitmapset64>*);

/* gpuqo_cpu_dpsub_bicc_parallel
 *
 *	 Parallel CPU baseline for GPU query optimization using the DP sub
 *   algorithm with BiCC optimization.
 */
template<typename BitmapsetN>
QueryTree<BitmapsetN>*
gpuqo_cpu_dpsub_bicc_parallel(GpuqoPlannerInfo<BitmapsetN>* info)
{
    DPsubBiCCCPUAlgorithm<BitmapsetN, level_hashtable<BitmapsetN>, true> alg;
    return gpuqo_cpu_dpsub_generic_parallel(info, &alg);
}

template QueryTree<Bitmapset32>* gpuqo_cpu_dpsub_bicc_parallel<Bitmapset32>(GpuqoPlannerInfo<Bitmapset32>*);
template QueryTree<Bitmapset64>* gpuqo_cpu_dpsub_bicc_parallel<Bitmapset64>(GpuqoPlannerInfo<Bitmapset64>*);