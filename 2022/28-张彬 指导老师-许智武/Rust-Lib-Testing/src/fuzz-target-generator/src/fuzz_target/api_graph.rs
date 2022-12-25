use crate::fuzz_target::api_function::ApiFunction;
use crate::fuzz_target::api_sequence::{ApiCall, ApiSequence, ParamType};
use crate::fuzz_target::api_util;
use crate::fuzz_target::call_type::CallType;
use crate::fuzz_target::fuzzable_type::{self, FuzzableType, FuzzableCallType};
use crate::fuzz_target::impl_util::*;
use crate::fuzz_target::mod_visibility::ModVisibity;
use crate::fuzz_target::prelude_type;
use crate::fuzz_target::file_util;
use crate::fuzz_target::operation_sequence::StatememtSrc;
use crate::fuzz_target::api_parameter::{self, ApiParameter, ApiStructure, GenericPath, PreludedStructure};
// use rustdoc::clean::{PrimitiveType};
use rand::{self, Rng};

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Lines};
use std::path::{Path, PathBuf};

pub use lazy_static::*;
use rustdoc::clean::{self, Visibility};
use rustdoc::clean::types::ItemId;
use rustc_hir::def_id::LocalDefId;
use std::env;

lazy_static! {
    static ref RANDOM_WALK_STEPS: HashMap<&'static str, usize> = {
        let mut m = HashMap::new();
        m.insert("regex", 10000);
        m.insert("url", 10000);
        m.insert("time", 10000);
        m
    };
}

lazy_static! {
    static ref CAN_COVER_NODES: HashMap<&'static str, usize> = {
        let mut m = HashMap::new();
        m.insert("regex", 96);
        m.insert("serde_json", 41);
        m.insert("clap", 66);
        m
    };
}

#[derive(Clone, Debug)]
pub struct ApiGraph {
    pub _crate_name: String,
    pub api_functions: Vec<ApiFunction>,
    pub api_functions_visited: Vec<bool>,
    pub api_dependencies: Vec<(ApiDependency, bool)>,
    pub api_sequences: Vec<ApiSequence>,
    pub api_parameters: Vec<ApiParameter>,
    pub full_name_map: FullNameMap,  //did to full_name
    pub mod_visibility: ModVisibity, //the visibility of mods，to fix the problem of `pub use`
    pub generic_functions: HashSet<String>,
    pub functions_with_unsupported_fuzzable_types: HashSet<String>,
    pub generable_types: HashMap<clean::Type, HashSet<(usize, CallType)>>,
    pub record_sequences: Vec<String>,              // 记录已有的seq序列（String形式）
    pub copy_structs: Vec<String>,
    pub stop_list: Vec<usize>,
    pub api_generation_map: HashMap<usize, Vec<ApiSequence>>,
    pub refuse_api_map: HashMap<usize, Vec<HashSet<usize>>>,
    pub type_generation_map: HashMap<String, Vec<ApiSequence>>,
    pub total_weight: usize,
    //pub _sequences_of_all_algorithm : FxHashMap<GraphTraverseAlgorithm, Vec<ApiSequence>>
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum GraphTraverseAlgorithm {
    BFS,
    FastBFS,
    BFSEndPoint,
    FastBFSEndPoint,
    RandomWalk,
    RandomWalkEndPoint,
    TryDeepBFS,
    DirectBackwardSearch,
    UnsafeSearch,
    UnsafeBFS,
    WeightedUnsafeBFS,
    PatternBased,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PatternOperation {
    Unsafe,
    GetRawPointer,
    Drop,
    Use,
    Mutate
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Copy)]
pub enum UnsafeSearch {
    All, // 所有序列拼接
    UnsafeParma, // 基于Unsafe参数相关拼接，相同struct的来源均一致
    OnlyUnsafeShare, // 只维持Unsafe相关的API来源一致
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Copy)]
pub enum ApiType {
    BareFunction,
    GenericFunction,
    ControlStart,
    ScopeEnd,
    UseParam,
    // GenericFunction, currently not support now
}

//函数的依赖关系
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ApiDependency {
    pub output_fun: (ApiType, usize), //the index of first func, usize就是apifunction的下标
    pub input_fun: (ApiType, usize),  //the index of second func
    pub input_param_index: usize,
    pub call_type: CallType,
    pub parameters: Vec<ApiParameter>,
    // pub generic_type: Option<String>, //仅当为input_fun.index为泛型参数时需要考虑
}

impl ApiDependency {
    // pub fn new(api_type: ApiType, call_func: usize, called_func: usize, param_index: uusize, call_type: CallType) {
    
    // }
    pub fn is_same_dependency(&self, other_dependency: &ApiDependency) -> bool {
        if self.output_fun == other_dependency.output_fun
        && self.input_fun == other_dependency.input_fun
        && self.input_param_index == other_dependency.input_param_index {
            return true;
        }
        // println!("{:?} - {:?} ", self.output_fun, other_dependency.output_fun);
        // println!("{:?} - {:?}", self.input_fun, other_dependency.input_fun);
        // println!("{:?} - {:?}", self.input_param_index, other_dependency.input_param_index);
        // println!("{:?}", self.call_type);
        return false;
    }
}

impl ApiGraph {
    pub fn new(_crate_name: &String) -> Self {
        let api_functions = Vec::new();
        let api_functions_visited = Vec::new();
        let api_dependencies = Vec::new();
        let api_sequences = Vec::new();
        let api_parameters = Vec::new();
        let full_name_map = FullNameMap::new();
        let mod_visibility = ModVisibity::new(_crate_name);
        let generic_functions = HashSet::new();
        let functions_with_unsupported_fuzzable_types = HashSet::new();
        let generable_types = HashMap::new();
        let record_sequences = Vec::new();
        let copy_structs = Vec::new();
        let stop_list = Vec::new();
        let api_generation_map = HashMap::new();
        let refuse_api_map = HashMap::new();
        let type_generation_map = HashMap::new();
        let total_weight = 0;
        //let _sequences_of_all_algorithm = FxHashMap::default();
        ApiGraph {
            _crate_name: _crate_name.clone(),
            api_functions,
            api_functions_visited,
            api_dependencies,
            api_sequences,
            api_parameters,
            full_name_map,
            mod_visibility,
            generic_functions,
            functions_with_unsupported_fuzzable_types,
            generable_types,
            record_sequences,
            copy_structs,
            stop_list, // 用于记录不可达的方法
            api_generation_map,
            refuse_api_map,
            type_generation_map,
            total_weight,
            //_sequences_of_all_algorithm,
        }
    }

    pub fn add_api_parameter(&mut self, api_param: ApiParameter) {
        // 已经有了就会替换
        match &api_param {
            ApiParameter::Preluded(..) => { 
                // 需要替换
                for param in &mut self.api_parameters {
                    if param.is_preluded() && param.as_string() == api_param.as_string() {
                        *param = api_param.clone();
                    }
                }
            },
            _ => {
                self.api_parameters.push(api_param);
            }
        }
    }

    pub fn add_api_function(&mut self, api_fun: ApiFunction) {
        if api_fun._is_generic_function() {
            self.generic_functions.insert(api_fun.full_name.clone());
            match env::var("GenericMode") {
                Ok(value) => {
                    match value.as_str() {
                        "true" => {
                            if !self.api_functions.contains(&api_fun) {
                                self.api_functions.push(api_fun);
                            }
                        }
                        _ => {}
                    }
                }
                Err(_) => {}
            }
        } else if api_fun.filter_by_fuzzable_type(&self.full_name_map) {
            self.functions_with_unsupported_fuzzable_types
                .insert(api_fun.full_name.clone());
        } else {
            if !self.api_functions.contains(&api_fun) {
                self.api_functions.push(api_fun);
            }
        }
        //println!("There are {} generic functions.", generic_functions.len());
        //println!("There are {} functions with unsupported fuzzable types.", functions_with_unsupported_fuzzable_types.len());
    }

    pub fn add_mod_visibility(&mut self, mod_name: &String, visibility: &Visibility) {
        self.mod_visibility.add_one_mod(mod_name, visibility);
    }

    pub fn filter_functions(&mut self) {
        self.filter_api_functions_by_prelude_type();
        self.filter_api_functions_by_mod_visibility();
    }

    pub fn filter_api_functions_by_prelude_type(&mut self) {
        let prelude_types = prelude_type::get_all_preluded_type();

        if prelude_types.len() <= 0 {
            return;
        }

        let mut new_api_functions = Vec::new();
        for api_func in &self.api_functions {
            let api_func_name = &api_func.full_name;
            let trait_name = &api_func._trait_full_path;
            let mut prelude_function_flag = false;
            for one_prelude_type in &prelude_types {
                if api_func_name
                    .as_str()
                    .starts_with(one_prelude_type.as_str())
                {
                    prelude_function_flag = true;
                    break;
                }
                if let Some(trait_name_) = trait_name {
                    if trait_name_.as_str().starts_with(one_prelude_type) {
                        prelude_function_flag = true;
                        break;
                    }
                }
            }
            if !prelude_function_flag {
                new_api_functions.push(api_func.clone());
            }
        }
        self.api_functions = new_api_functions;
    }

    pub fn filter_api_functions_by_mod_visibility(&mut self) {
        let invisible_mods = self.mod_visibility.get_invisible_mods();

        if invisible_mods.len() <= 0 {
            return;
        }

        let mut new_api_functions = Vec::new();
        for api_func in &self.api_functions {
            let api_func_name = &api_func.full_name;
            let trait_name = &api_func._trait_full_path;
            let mut invisible_flag = false;
            for invisible_mod in &invisible_mods {
                if api_func_name.as_str().starts_with(invisible_mod.as_str()) {
                    invisible_flag = true;
                    break;
                }
                if let Some(trait_name_) = trait_name {
                    if trait_name_.as_str().starts_with(invisible_mod) {
                        invisible_flag = true;
                        break;
                    }
                }
            }
            if !invisible_flag {
                new_api_functions.push(api_func.clone());
            }
        }
        self.api_functions = new_api_functions;
    }

    pub fn set_full_name_map(&mut self, full_name_map: &FullNameMap) {
        self.full_name_map = full_name_map.clone();
    }

    pub fn find_all_dependencies(&mut self) {
        //println!("find_dependencies");
        self.api_dependencies.clear();
        //两个api_function之间的dependency
        let api_num = self.api_functions.len();
        println!("\n>>>>> FIND ALL DEPENDENCIES");
        for i in 0..api_num {
            let first_fun = &self.api_functions[i].clone();
            let first_api_type = match first_fun._is_generic_function() {
                true => { ApiType::GenericFunction }
                false => { ApiType::BareFunction }
            };
            if first_fun._is_end_function(&self.full_name_map) {
                //如果第一个函数是终止节点，就不寻找这样的依赖
                continue;
            }
            if let Some(ty_) = &first_fun.output {
                let output_type = ty_;
                for j in 0..api_num {
                    //TODO:是否要把i=j的情况去掉？
                    let second_fun = &self.api_functions[j].clone();
                    if second_fun._is_start_function(&self.full_name_map) {
                        //如果第二个节点是开始节点，那么直接跳过
                        continue;
                    }
                    let second_api_type = match second_fun._is_generic_function() {
                        true => { ApiType::GenericFunction }
                        false => { ApiType::BareFunction }
                    };
                    let input_params = &second_fun.inputs;
                    let input_params_num = input_params.len();
                    for k in 0..input_params_num {
                        // let input_param = self.get_real_type(second_fun, &input_params[k]);
                        let input_param = &input_params[k];
                        // println!("output: {:#?}", output_type);
                        // println!("input: {:#?}", input_param);
                        // 判断call type
                        let call_type = api_util::_same_type(
                            output_type,
                            &input_param,
                            true,
                            &self.full_name_map,
                        );
                        // TODO: CallType::_Deref的时候要考虑数据类型是否能copy
                        match &call_type {
                            CallType::_NotCompatible => {
                                println!("FAILED: {}->{}-{}", first_fun.full_name, second_fun.full_name, k);
                                continue;
                            }
                            _ => {
                                if let Some(parameters) = self.match_generic_type(first_fun, second_fun, k, &call_type) {
                                    let one_dependency = ApiDependency {
                                        output_fun: (first_api_type, i),
                                        input_fun: (second_api_type, j),
                                        input_param_index: k,
                                        call_type: call_type.clone(),
                                        parameters: parameters,
                                    };
                                    let is_unsafe = first_fun.is_unsafe_function() || second_fun.is_unsafe_function();
                                    self.api_dependencies.push((one_dependency, is_unsafe));
                                    
                                    let need_function = self.generable_types.entry(input_param.clone()).or_insert(HashSet::new());
                                    need_function.insert((i, call_type.clone()));
                                    // let api_functions = &mut self.api_functions;
                                    self.api_functions[i].update_next_function(k, j);
                                    self.api_functions[j].update_need_function(k, i);
                                    println!("SUCCESS: {}->{}-{}", i, j, k);
                                } else {
                                    println!("FAILED: {}->{}-{}", i, j, k);
                                }
                            }
                        }
                    }
                }
            }
        }
        println!("FIND DEPENDENCIES FINISHED <<<<<\n");
    }

    pub fn default_generate_sequences(&mut self) {
        //BFS + backward search
        self.generate_all_possible_sequences(GraphTraverseAlgorithm::BFSEndPoint);
        self._try_to_cover_unvisited_nodes();

        // backward search
        //self.generate_all_possible_sequences(GraphTraverseAlgorithm::_DirectBackwardSearch);
    }

    // author：张彬
    pub fn dfs_generate_sequences(&mut self) {
        // DFS + backword search
        self.generate_all_possible_sequences_dfs();
        self._try_to_cover_unvisited_nodes();
    }

    // author：张彬
    pub fn dfs_with_start_generate_sequence(&mut self) {
        // DFS + backword search
        self.generate_all_possible_sequences_dfs_with_start();
        self._try_to_cover_unvisited_nodes();
    }

    pub fn unsafe_bfs_generate_sequences(&mut self) {
        //unsafeBFS + backward search
        self.generate_all_possible_sequences(GraphTraverseAlgorithm::UnsafeBFS);
        self._try_to_cover_unvisited_nodes();
    }

    pub fn weighted_unsafe_bfs_generate_sequences(&mut self) {
        //unsafeBFS + backward search
        self.generate_all_possible_sequences(GraphTraverseAlgorithm::WeightedUnsafeBFS);
        self._try_to_cover_unvisited_nodes();
    }

    pub fn unsafe_generate_sequences(&mut self) {
        // self.unsafe_based_search(UnsafeSearch::UnsafeParma);
        self.unsafe_based_search(UnsafeSearch::OnlyUnsafeShare);
        self.update_dependencies();
    }

    pub fn pattern_based_generate_sequences(&mut self) {
        self.generate_all_possible_sequences(GraphTraverseAlgorithm::PatternBased);
    }
    
    pub fn update_dependencies(&mut self) {
        let graph_ = self.clone();
        for sequence in &mut self.api_sequences {
            // println!("seq: {}", sequence);
            let sequnece_ = sequence.clone();
            for api_call in &sequnece_.functions {
                let (api_type, func_index) = api_call.func;
                let called_function = &self.api_functions[func_index];
                let is_unsafe_called_func = called_function.is_unsafe_function();
                for i in 0..api_call.params.len() {
                    let param = &api_call.params[i];
                    let (type_, position_, call_type_) = param;
                    match type_ {
                        ParamType::_FunctionReturn => {
                            let input_api_call = &sequnece_.functions[*position_];
                            let (need_api_type, need_func_index) = input_api_call.func;
                            let is_unsafe_need_func = &self.api_functions[need_func_index].is_unsafe_function();
                            // println!("dependency: {}->{}", need_func_index, func_index);
                            if let Some(dependency_index) = graph_.check_dependency(&need_api_type, need_func_index, &api_type, func_index, i) {
                                sequence._add_dependency((dependency_index, is_unsafe_called_func || *is_unsafe_need_func));
                            }
                        },      
                        _ => {}
                    }
                }
            }
        }
    }

    pub fn generate_all_possible_sequences(&mut self, algorithm: GraphTraverseAlgorithm) {
        //BFS序列的最大长度：即为函数的数量,或者自定义
        //let bfs_max_len = self.api_functions.len();
        let bfs_max_len = 3;
        let bfs_max_len = match env::var("bfs_len"){
            Ok(value) => value.parse::<usize>().unwrap(),
            Err(_) => 3,
        };
        //random walk的最大步数

        let random_walk_max_size = if RANDOM_WALK_STEPS.contains_key(self._crate_name.as_str()) {
            RANDOM_WALK_STEPS
                .get(self._crate_name.as_str())
                .unwrap()
                .clone()
        } else {
            100000
        };

        //no depth bound
        let random_walk_max_depth = 0;
        //try deep sequence number
        let max_sequence_number = 100000;
        match algorithm {
            GraphTraverseAlgorithm::BFS => {
                println!("using bfs");
                self.bfs(bfs_max_len, false, false);
            }
            GraphTraverseAlgorithm::FastBFS => {
                println!("using fastbfs");
                self.bfs(bfs_max_len, false, true);
            }
            GraphTraverseAlgorithm::BFSEndPoint => {
                println!("using bfs end point");
                self.bfs(bfs_max_len, true, false);
            }
            GraphTraverseAlgorithm::FastBFSEndPoint => {
                println!("using fast bfs end point");
                self.bfs(bfs_max_len, true, true);
            }
            GraphTraverseAlgorithm::TryDeepBFS => {
                println!("using try deep bfs");
                self._try_deep_bfs(max_sequence_number);
            }
            GraphTraverseAlgorithm::RandomWalk => {
                println!("using random walk");
                self.random_walk(random_walk_max_size, false, random_walk_max_depth);
            }
            GraphTraverseAlgorithm::RandomWalkEndPoint => {
                println!("using random walk end point");
                self.random_walk(random_walk_max_size, true, random_walk_max_depth);
            }
            GraphTraverseAlgorithm::DirectBackwardSearch => {
                println!("using backward search");
                self.api_sequences.clear();
                self.reset_visited();
                self._try_to_cover_unvisited_nodes();
            }
            GraphTraverseAlgorithm::UnsafeSearch => {
                println!("using unsafe search");
                self.unsafe_generate_sequences();
            }
            GraphTraverseAlgorithm::UnsafeBFS => {
                println!("using unsafe search");
                self.unsafe_bfs(bfs_max_len, true, false);
            }
            GraphTraverseAlgorithm::WeightedUnsafeBFS => {
                println!("using weighted unsafe search");
                self.weighted_unsafe_bfs();
            }
            GraphTraverseAlgorithm::PatternBased => {
                println!("using pattern based search");
                self.pattern_based_search();
            }
        }
    }

    // author：张彬
    pub fn generate_all_possible_sequences_dfs(&mut self) {
        let dfs_max_len = 3;
        let dfs_max_len = match env::var("dfs_len"){
            Ok(value) => value.parse::<usize>().unwrap(),
            Err(_) => 3,
        };

        println!("using dfs");
        self.dfs(3, false);
    }

    // author：张彬
    pub fn generate_all_possible_sequences_dfs_with_start(&mut self) {
        let dfs_max_len = 3;
        let dfs_max_len = match env::var("dfs_start_len"){
            Ok(value) => value.parse::<usize>().unwrap(),
            Err(_) => 3,
        };

        println!("using dfs_with_start");
        self.dfs_with_start(3);
    }

    pub fn reset_visited(&mut self) {
        self.api_functions_visited.clear();
        let api_function_num = self.api_functions.len();
        for _ in 0..api_function_num {
            self.api_functions_visited.push(false);
        }
        //TODO:还有别的序列可能需要reset
    }

    //检查是否所有函数都访问过了
    pub fn check_all_visited(&self) -> bool {
        let mut visited_nodes = 0;
        for visited in &self.api_functions_visited {
            if *visited {
                visited_nodes = visited_nodes + 1;
            }
        }

        if CAN_COVER_NODES.contains_key(self._crate_name.as_str()) {
            let to_cover_nodes = CAN_COVER_NODES
                .get(self._crate_name.as_str())
                .unwrap()
                .clone();
            if visited_nodes == to_cover_nodes {
                return true;
            } else {
                return false;
            }
        }

        if visited_nodes == self.api_functions_visited.len() {
            return true;
        } else {
            return false;
        }
    }

    //已经访问过的节点数量,用来快速判断bfs是否还需要run下去：如果一轮下来，bfs的长度没有发生变化，那么也可直接quit了
    pub fn _visited_nodes_num(&self) -> usize {
        let visited: Vec<&bool> = (&self.api_functions_visited)
            .into_iter()
            .filter(|x| **x == true)
            .collect();
        visited.len()
    }
    
    pub fn is_reachable(&self, func_index: usize, i: usize) -> Option<Vec<(usize, CallType)>> {
        let mut reachable_functions = Vec::new();
        let target_ty = &self.api_functions[func_index].inputs[i];
        for (ty_, function_set_) in &self.generable_types.clone() {
            for (function_index_, call_type_) in function_set_ {
                if ty_ == target_ty {
                    reachable_functions.push((*function_index_, call_type_.clone()));
                } else {
                    let call_type = api_util::_same_type(
                        ty_,
                        target_ty,
                        true,
                        &self.full_name_map,
                    );
                    match &call_type {
                        CallType::_NotCompatible => {continue;},
                        _ => {
                            reachable_functions.push((*function_index_, call_type.clone()));
                            // 想更新self.generable_types但是需要mut,会导致其它问题
                            // let reachable_function_set = self.generable_types.entry(target_ty.clone()).or_insert(HashSet::new());
                            // reachable_function_set.insert((*function_index_, call_type.clone()));
                        }
                    }
                }
            }
        }
        if reachable_functions.len() == 0 {
            return None;
        } else {
            return Some(reachable_functions)
        }
    }

    pub fn get_reachable_dependencies(&self, func_index: usize, param_index: usize) -> Option<Vec<ApiDependency>> {
        let mut reachable_dependecies = Vec::new();
        let api_type = match self.api_functions[func_index]._is_generic_function() {
            true => ApiType::GenericFunction,
            false => ApiType::BareFunction,
        };
        for i in 0..self.api_dependencies.len() {
            let dependency = &self.api_dependencies[i].0;
            if dependency.input_fun == (api_type, func_index) 
            && dependency.input_param_index == param_index {
                reachable_dependecies.push(dependency.clone());
            }
        }
        if reachable_dependecies.len() > 0 {
            return Some(reachable_dependecies);
        }
        None
    }

    pub fn insert_unvisited_api_seq(
        &mut self, 
        unvisited_sequences: &HashMap<usize, Vec<ApiSequence>>,
        merged_seq: &ApiSequence, 
        create_index: usize, 
        unsafe_index1: usize, 
        unsafe_index2: usize,
        unsafe_ty: &String,
    ) {
        let mut insert_flag = false;
        for (index, sequences) in unvisited_sequences {
            // 查看unsafe struct是否一致
            let mut flag = false;
            let mut input_index = 0;
            for i in 0..self.api_functions[*index].input_types.len() {
                let type_ = &self.api_functions[*index].input_types[i];
                // 如果unsafe类型有一样的，才插入
                // let mut naked_type = type_.clone();
                // if naked_type.starts_with("&mut") {
                //     naked_type = naked_type.chars().skip(5).collect();
                //     // naked_type = split_str[1];
                // } else if naked_type.starts_with('&') {
                //     naked_type = naked_type.chars().skip(1).collect();
                // } else if 
                let naked_type = get_naked_type(&type_.to_string());
                // println!("{} ---- {}", unsafe_ty, naked_type);
                if *unsafe_ty == naked_type {
                    flag = true;
                    input_index = i;
                    break;
                }
            }
            // println!("Inserting...{}", flag);
            if !flag {
                continue;
            } else {
                insert_flag = true;
            }
            // 开始插入序列
            for seq in sequences {
                let mut res_seq = merged_seq.clone();
                // 获得call type
                let offer_function_index = res_seq.functions[create_index].func.1;
                // let unvisited_function_index = seq.functions[seq.functions.len()-1].func.1;
                if let Some(output_type) = &self.api_functions[offer_function_index].output{
                    let input_param = &self.api_functions[*index].inputs[input_index];
                    let call_type = api_util::_same_type(&output_type, &input_param, true, &self.full_name_map);
                    // 先插入后面的序列，这样不会影响前面的序列。
                    match &call_type {
                        CallType::_NotCompatible => {println!("Something wrong in insert unvisited");}
                        // 如果该api不存在影响的变量的情况则不插入，需要在这个地方进行细化
                        _ => {
                            if api_util::_need_mut_tag(&call_type) {
                                // println!("mut tag for {}", create_index);
                                res_seq._insert_function_mut_tag(create_index);
                            }
                            // println!("Inserting Seq: {}\nCall Type: {:#?}", seq, call_type);
                            // res_seq._insert_another_sequence(&seq, unsafe_index2 + 1, create_index, input_index, *index, &call_type);
                            // res_seq._insert_another_sequence(&seq, unsafe_index1 + 1, create_index, input_index, *index, &call_type);
                            res_seq._insert_another_sequence(&seq, unsafe_index1, create_index, input_index, *index, &call_type);
                        }
                    }
                    // println!("Inserted Seq: {}", res_seq);
                    if !self.record_sequences.contains(&res_seq.get_sequence_string()){
                        // println!("Inserted Seq: {:#?}", res_seq);
                        self.record_sequences.push(res_seq.get_sequence_string().clone());
                        if self.check_syntax(&res_seq){
                            res_seq.delete_redundant_fuzzable_params();
                            res_seq.add_mut_flag();
                            self.api_sequences.push(res_seq);
                        }
                    }
                }
            }
        }
        // 如果unvisit都不满足插入，则不插入
        if !insert_flag {
            println!("insert flag false");
            if self.check_syntax(&merged_seq) {
                let mut res_seq = merged_seq.clone();
                res_seq.delete_redundant_fuzzable_params();
                res_seq.add_mut_flag();
                self.api_sequences.push(res_seq.clone());
            }
        }
    }

    pub fn unsafe_based_search(&mut self, unsafe_strategy: UnsafeSearch) {
        let mut unvisited_indexes = Vec::new();
        let api_function_num = self.api_functions.len();
        let api_sequence = ApiSequence::new();
        let mut unsafe_struct_map: HashMap<String, Vec<usize>> = HashMap::new();

        self.api_sequences.clear();
        self.reset_visited();
        //无需加入长度为1的，从空序列开始即可，加入一个长度为0的序列作为初始
        // self.api_sequences.push(api_sequence);

        // 构建 <struct, api> map
        let mut unsafe_indexes = Vec::new();
        for i in 0..self.api_functions.len() {
            let func = &self.api_functions[i];
            if func.is_unsafe_function() {
                unsafe_indexes.push(i);
                for func_type in &func.func_types {
                    if let Some(api_indexes) = unsafe_struct_map.get_mut(func_type) {
                        api_indexes.push(i);
                    } else {
                        unsafe_struct_map.insert(func_type.to_string(), Vec::from([i,]));
                    }
                }
            }
        }

        println!("Crate Name: {:?}", self._crate_name);
        if unsafe_indexes.len() == 0 {
            println!("There is no unsafe API");
            return
        }
        
        // 寻找unsafe API并生成相应的需求序列（即产生能到达该UNSAFE API的序列)
        let mut unsafe_sequences_map: HashMap<usize, Vec<ApiSequence>> = HashMap::new();
        for unsafe_index in &unsafe_indexes {
            println!("Unsafe index: {}", unsafe_index);
            if let Some(mut unsafe_sequences) = self.reverse_collect_unsafe_sequence(*unsafe_index){
                // 1. 将生成的序列简化（合并相同的param）
                for unsafe_seq in &mut unsafe_sequences {
                    println!("Unsafe Generation Sequence: {}", unsafe_seq);
                    // 这里不做删除冗余的操作，此时所有unsafe的inputs都独立生成
                    // unsafe_seq.remove_duplicate_param();
                    for f in &unsafe_seq.functions {
                        // 更新 Api visited 情况
                        self.api_functions_visited[f.func.1] = true;
                    }
                }
                // 更新 visited api
                unsafe_sequences_map.insert(*unsafe_index, unsafe_sequences);
            }
        }
        // println!("unsafe sequence generation done.\n{:#?}", unsafe_sequences_map);
        println!("Total Unsafe API Number: {}", unsafe_indexes.len());
        println!("Reachable Unsafe API Number: {}", unsafe_sequences_map.keys().len());
        println!("Unsafe API Coverage: {}", unsafe_sequences_map.keys().len() / unsafe_indexes.len());
        
        // 未访问到的functions
        let mut unvisited_sequences_map: HashMap<usize, Vec<ApiSequence>> = HashMap::new();
        for i in 0..self.api_functions_visited.len() {
            if !self.api_functions_visited[i] {
                let related_structs = &self.api_functions[i].input_types;
                let mut related_flag = false;
                for related_struct in related_structs {
                    let related_simple_struct = facilitate_type_name(&related_struct.to_string());
                    if let Some(_) = unsafe_struct_map.get(&related_simple_struct) {
                        related_flag = true;
                        break;
                    }
                }
                if !related_flag {
                    println!("Filter Unvisited API: {:?}", i);
                    continue;
                }

                println!("Potential Unvisited API: {:?}", i);
                unvisited_indexes.push(i);
                if let Some(unvisited_sequences) = self.reverse_search_sequence(i, 0) {
                    // for unvisited_seq in &mut unvisited_sequences {
                        // println!("before remove duplicate: {:#?}", unvisited_seq);
                        // unvisited_seq.remove_duplicate_param();
                        // unvisited_seq._form_control_block();
                        // println!("after remove duplicate: {:#?}", unvisited_seq);
                    // }
                    // 生成未访问到的API的完整序列
                    // 合并的时候要注意unsafe的来源须和unsafe sequence的统一
                    unvisited_sequences_map.insert(i, unvisited_sequences);
                }
            }
        }
        println!("unvisited sequence generation done.");

        println!("start merge");
        match unsafe_strategy {
            UnsafeSearch::All => {
                for (key1, sequences1) in &unsafe_sequences_map {
                    for (key2, sequences2) in &unsafe_sequences_map {
                        // 判断序列是否相同
                        if key1 == key2 { continue; }
                        for seq1 in sequences1 {
                            for seq2 in sequences2 {
                                // 判断序列是否有包含关系，如果有则跳过

                                // if seq1.is_contain(&seq2) | seq2.is_contain(&seq1) {
                                //     println!("Contain Relationship: {}, {}", seq1.get_sequence_string(), seq2.get_sequence_string());
                                //     continue;
                                // }

                                let mut merged_seq = seq1._merge_another_sequence(seq2);
                                merged_seq.remove_duplicate_param();
                                self.api_sequences.push(merged_seq);
                            }
                        }
                    }
                }
            }
            UnsafeSearch::UnsafeParma => {
                // 如果unsafe涉及到的struct type一样，则配对，配对没有先后关系
                let unsafe_pairs = self.get_unsafe_pair(&unsafe_indexes);
                println!("Unsafe Based Parmater");
                println!("{:?}", unsafe_pairs);
                for (i1, i2, types) in &unsafe_pairs {
                    let sequences1 = match unsafe_sequences_map.get(i1){
                        None => continue,
                        Some(_) => unsafe_sequences_map.get(i1).unwrap()
                    };
                    let sequences2 = match unsafe_sequences_map.get(i2){
                        None => continue,
                        Some(_) => unsafe_sequences_map.get(i2).unwrap()
                    };
                    for seq1 in sequences1 {
                        for seq2 in sequences2 {
                            // 判断序列是否有包含关系A包含于B，则只取B
                            if seq1.is_contain(seq2) | seq2.is_contain(seq1) {
                                println!("Contain Relationship: {}, {}", seq1.get_sequence_string(), seq2.get_sequence_string());
                                continue;
                            }
                
                            let mut merged_seq1 = seq1._merge_another_sequence(seq2);
                            merged_seq1.remove_duplicate_param();
                            if self.check_syntax(&merged_seq1) {
                                self.api_sequences.push(merged_seq1);
                            }
                            let mut merged_seq2 = seq2._merge_another_sequence(seq1);
                            merged_seq2.remove_duplicate_param();
                            if self.check_syntax(&merged_seq2) {
                                self.api_sequences.push(merged_seq2);
                            }
                        }
                    }
                }
            }
            UnsafeSearch::OnlyUnsafeShare => {
                // 合并Unsafe Sequence
                // 1. 查询Unsafe Function涉及的Unsafe Struct Type(目前能找到RawPtr)
                // 2. 若两个涉及到的Unsafe Struct Type一致，则进行合并
                // 3. 合并分为两种情况，(seq2,删减的seq1)以及(seq1,删减的seq2)，删减意味着把unsafe struct type相关的生成序列删除，让merged seq的unsafe来源一致
                // 注：要考虑Call type的更新情况，而不是直接链接。
                
                let unsafe_pairs = self.get_unsafe_pair(&unsafe_indexes); // 如果unsafe涉及到的struct type一样，则配对，配对没有先后关系
                println!("Only Unsafe Share");
                // println!("Unsafe Pair: {:#?}", unsafe_pairs);

                // 记录未使用到的unsafe api生成序列，作为结果序列生成。
                // 格式为<int, int>，前一个int代表unsafe api，后一个代表的是该api的第几个生成序列
                let mut record_seq_set = HashSet::new();
                // let mut merged_seqs = Vec::new(); // 用于记录合并成过的seq，避免重复
                for (i1, i2, types) in &unsafe_pairs {
                    let sequences1 = match unsafe_sequences_map.get(i1){
                        None => continue,
                        Some(_) => unsafe_sequences_map.get(i1).unwrap()
                    };
                    let sequences2 = match unsafe_sequences_map.get(i2){
                        None => continue,
                        Some(_) => unsafe_sequences_map.get(i2).unwrap()
                    };
                    for j1 in 0..sequences1.len() {
                        for j2 in 0..sequences2.len() {
                            let seq1 = &sequences1[j1];
                            let seq2 = &sequences2[j2];
                            // 长度为一的序列好像没必要合并？
                            if seq1.functions.len()==1 || seq2.functions.len()==1 {
                                continue;
                            }
                            // println!("Merging Seq1: {}, Seq2: {}", seq1, seq2);
                            // if *i1 == 9 && *i2 == 35 {
                            //     println!("seq1: {}", seq1);
                            //     println!("{:#?}", seq1);
                            //     println!("seq1: {}", seq2);
                            //     println!("{:#?}", seq2);
                            // }
                            // 判断序列是否有包含关系A包含于B，则只取B
                            if seq1.is_contain(seq2) | seq2.is_contain(seq1) {
                                println!("Contain Relationship: {}, {}", seq1.get_sequence_string(), seq2.get_sequence_string());
                                continue;
                            }
                            for ty in types {
                                // 根据unsafe struct寻找相应的input编号
                                let mut input_index1 = 0;
                                let mut input_index2 = 0;
                                let (mut flag1, mut flag2) = (false, false);
                                let input_types1 = &self.api_functions[*i1].input_types;
                                for j in 0..input_types1.len(){
                                    let mut ty1 = input_types1[j].clone();
                                    ty1 = facilitate_type_name(&ty1);
                                    if *ty == ty1{
                                        input_index1 = j;
                                        flag1 = true;
                                    }
                                }

                                let input_types2 = &self.api_functions[*i2].input_types;
                                for j in 0..input_types2.len() {
                                    let mut ty2 = input_types2[j].clone();
                                    ty2 = facilitate_type_name(&ty2);
                                    if *ty == ty2 {
                                        input_index2 = j;
                                        flag2 = true;
                                    }
                                }
                                // println!("flag1: {}, flag2: {}", flag1, flag2);
                                if flag1 | flag2 { 
                                    record_seq_set.insert((i1, j1));
                                    record_seq_set.insert((i2, j2)); 
                                }
                                // 按照编号进行合并
                                if flag1 {
                                    if let Some(removed_seq1) = seq1.remove_input_create_api(input_index1, 0){
                                        println!("merging with input: {}", input_index1);
                                        let mut merged_seq = seq2._merge_another_sequence(&removed_seq1);
                                        println!("merged seq: {}", merged_seq);
                                        // 寻找seq2中，生成unsafe的function
                                        let seq2_len = seq2.functions.len();
                                        let merged_seq_len = merged_seq.functions.len();
                                        let offer_input_index = merged_seq.functions[seq2_len-1].params[input_index2].1;
                                        // 与removed seq建立联系
                                        let offer_function_index = merged_seq.functions[offer_input_index].func.1;
                                        if let Some(output_type) = &self.api_functions[offer_function_index].output{
                                            println!("output type: {:?}", output_type);
                                            let input_param = &self.api_functions[*i1].inputs[input_index1];
                                            let call_type = api_util::_same_type(&output_type, &input_param, true, &self.full_name_map);
                                            match &call_type {
                                                CallType::_NotCompatible => {
                                                    println!("Something wrong in connect unsafe");
                                                }
                                                _ => {
                                                    merged_seq.functions[merged_seq_len-1].params[input_index1] = 
                                                        (
                                                            ParamType::_FunctionReturn,
                                                            offer_input_index,
                                                            call_type.clone(),
                                                        );
                                                    // merge sequence 可能会重复
                                                    // TODO:(选取fuzzable params最少的那个)
                                                    println!("merged seq: {}", merged_seq);
                                                    if !self.record_sequences.contains(&merged_seq.get_sequence_string()){
                                                        // println!("merged seq1: {}", merged_seq);
                                                        // println!("{:#?}", merged_seq);
                                                        self.insert_unvisited_api_seq(&unvisited_sequences_map, &merged_seq, offer_input_index, seq2_len-1, merged_seq_len-1, &ty);
                                                        self.record_sequences.push(merged_seq.get_sequence_string().clone());
                                                    }
                                                }
                                            }
                                            
                                            // println!("Merged seq: {}", merged_seq);
                                            // self.api_sequences.push(merged_seq);
                                        }
                                    }
                                }
                                if flag2 {
                                    if let Some(removed_seq2) = seq2.remove_input_create_api(input_index2, 0){
                                        let mut merged_seq = seq1._merge_another_sequence(&removed_seq2);
                                        // 寻找seq2中，生成unsafe的function
                                        let seq1_len = seq1.functions.len();
                                        let merged_seq_len = merged_seq.functions.len();
                                        let offer_input_index = merged_seq.functions[seq1_len-1].params[input_index1].1;
                                        // 与removed seq建立联系
                                        let offer_function_index = merged_seq.functions[offer_input_index].func.1;
                                        if let Some(output_type) = &self.api_functions[offer_function_index].output{
                                            let input_param = &self.api_functions[*i2].inputs[input_index2];
                                            let call_type = api_util::_same_type(&output_type, &input_param, true, &self.full_name_map);
                                            match &call_type {
                                                CallType::_NotCompatible => {
                                                    println!("Something wrong in connect unsafe");
                                                }
                                                _ => {
                                                    merged_seq.functions[merged_seq_len-1].params[input_index2] = 
                                                        (
                                                            ParamType::_FunctionReturn,
                                                            offer_input_index,
                                                            call_type.clone(),
                                                        );
                                                    if !self.record_sequences.contains(&merged_seq.get_sequence_string()){
                                                        self.insert_unvisited_api_seq(&unvisited_sequences_map, &merged_seq, offer_input_index, seq1_len-1, merged_seq_len-1, &ty);
                                                        self.record_sequences.push(merged_seq.get_sequence_string().clone());
                                                    }
                                                }
                                            }
                                            // println!("Merged seq: {}", merged_seq);
                                            // self.api_sequences.push(merged_seq);
                                        }
                                    }
                                }
                                if !flag1 && !flag2 {
                                    let seq1_len = seq1.functions.len();
                                    let seq2_len = seq2.functions.len();
                                    if !self.record_sequences.contains(&seq1.get_sequence_string()){
                                        self.insert_unvisited_api_seq(&unvisited_sequences_map, &seq1, seq1.functions[seq1_len-1].params[input_index1].1, seq1_len-1, seq1_len-1, &ty);
                                        self.record_sequences.push(seq1.get_sequence_string().clone());
                                    }
                                    if !self.record_sequences.contains(&seq2.get_sequence_string()){
                                        self.insert_unvisited_api_seq(&unvisited_sequences_map, &seq2, seq2.functions[seq2_len-1].params[input_index2].1, seq2_len-1, seq2_len-1, &ty);
                                        self.record_sequences.push(seq2.get_sequence_string().clone());
                                    }
                                }
                            }
                        }
                    }
                }

                // 将未使用到的unsafe API生成序列作为结果序列：
                // 可选择是否需要
                for i in &unsafe_indexes {
                    if let Some(sequences) = unsafe_sequences_map.get(i) {
                        for j in 0..sequences.len() {
                            if !record_seq_set.contains(&(i,j)){
                                if self.check_syntax(&sequences[j]){
                                    self.api_sequences.push(sequences[j].clone());
                                }
                            }
                        }
                    }
                }
            }
        }
        // 插入unvisited且和unsafe struct相关的API seq
        // 
    }

    pub fn unsafe_based_search2(&mut self, unsafe_strategy: UnsafeSearch) {
        // 找到unsafe api -> 按struct分类
        // <struct, api>
        // <api, api sequence>

        // 找到potential api -> 按struct分类(能影响值)
        // <struct api>
        // <api, api sequence>
        
        println!("Crate Name: {:?}", self._crate_name);

        let mut unsafe_struct_map: HashMap<String, Vec<usize>> = HashMap::new();
        // let mut potential_struct_map: HashMap<String, Vec<usize>> = HashMap::new();
        
        // 重置
        self.api_sequences.clear();
        self.reset_visited();

        // 记录所有unsafe api index
        // 构建 <struct, api> map
        let mut unsafe_indexes = Vec::new();
        for i in 0..self.api_functions.len() {
            let func = &self.api_functions[i];
            if func.is_unsafe_function() {
                unsafe_indexes.push(i);
                for func_type in &func.func_types {
                    if let Some(api_indexes) = unsafe_struct_map.get_mut(func_type) {
                        api_indexes.push(i);
                    } else {
                        unsafe_struct_map.insert(func_type.to_string(), Vec::from([i,]));
                    }
                }
            }
        }
        println!("Total {} Unsafe API: {:?}", unsafe_indexes.len(), unsafe_indexes);
        println!("Unsafe Struct Map: {:#?}", unsafe_struct_map);
        if unsafe_indexes.len() == 0 {
            println!("There is no unsafe API");
            return
        }

        // 寻找unsafe API并生成相应的需求序列（即产生能到达该UNSAFE API的序列)
        // 并按数据结构构建<strcut, api>
        println!("get unsafe sequence");
        let mut unsafe_sequences_map: HashMap<usize, Vec<ApiSequence>> = HashMap::new();
        let mut merge_map: HashMap<usize, HashSet<(usize, usize)>> = HashMap::new();
        for (struct_type, unsafe_indexes) in unsafe_struct_map {
            let mut generated_indexes = Vec::new();
            for unsafe_index in &unsafe_indexes {
                if generated_indexes.contains(unsafe_index) {
                    println!("Unsafe index: {} already generated.", unsafe_index);
                    continue;
                }
                println!("Unsafe index: {}", unsafe_index);
                if let Some(mut unsafe_sequences) = self.reverse_collect_unsafe_sequence(*unsafe_index){
                    // 1. 将生成的序列简化（合并相同的param）
                    for unsafe_seq in &mut unsafe_sequences {
                        println!("Unsafe Generation Sequence: {}", unsafe_seq);
                        // 这里不做删除冗余的操作，此时所有unsafe的inputs都独立生成
                        // unsafe_seq.remove_duplicate_param();
                        for f in &unsafe_seq.functions {
                            // 更新 Api visited 情况
                            self.api_functions_visited[f.func.1] = true;
                        }
                    }
                    // 更新 visited api
                    unsafe_sequences_map.insert(*unsafe_index, unsafe_sequences);
                    // 更新 generated indexes
                    generated_indexes.push(*unsafe_index);
                }
            }

            if unsafe_indexes.len() < 2 { continue; }
            // 更新merge map
            for i in 0..generated_indexes.len() {
                let mut indexes_ = generated_indexes.clone();
                // 要不要考虑自己和自己合并？remove掉了就是不考虑
                indexes_.remove(i);
                if let Some(target_indexes) = merge_map.get_mut(&generated_indexes[i]) {
                    for index in indexes_ {
                        let input_types = &self.api_functions[index].input_types;
                        // 获取对应的输入序号
                        for j in 0..input_types.len(){
                            let mut ty = input_types[j].clone();
                            ty = facilitate_type_name(&ty);
                            if *struct_type == ty{
                                target_indexes.insert((index, j));
                            }
                        }
                    }
                } else {
                    let mut target_indexes = HashSet::new();
                    for index in indexes_ {
                        let input_types = &self.api_functions[index].input_types;
                        // 获取对应的输入序号
                        for j in 0..input_types.len(){
                            let mut ty = input_types[j].clone();
                            ty = facilitate_type_name(&ty);
                            if *struct_type == ty{
                                target_indexes.insert((index, j));
                            }
                        }
                    }
                    merge_map.insert(generated_indexes[i], target_indexes);
                }
            }
        }
        println!("Merge Map: {:#?}", merge_map);

        // 开始合并：
        // 在api上拼接api2（以api1为基准）
        // for (api1, (api2, input_index2)) in merge_map {
        //     sequences1 = unsafe_sequences_map.get(api1);
        //     sequences2 = unsafe_sequences_map.get(api2);
        //     for j1 in 0..sequences1.len() {
        //         for j2 in 0..sequences2.len() {
        //             let seq1 = &sequences1[j1];
        //             let seq2 = &sequences2[j2];
        //             if let Some(removed_seq1) = seq1.remove_input_create_api(input_index1, 0){
        //                 println!("merging with input: {}", input_index1);
        //                 let mut merged_seq = seq2._merge_another_sequence(&removed_seq1);
        //                 println!("merged seq: {}", merged_seq);
        //                 // 寻找seq2中，生成unsafe的function
        //                 let seq2_len = seq2.functions.len();
        //                 let merged_seq_len = merged_seq.functions.len();
        //                 let offer_input_index = merged_seq.functions[seq2_len-1].params[input_index2].1;
        //                 // 与removed seq建立联系
        //                 let offer_function_index = merged_seq.functions[offer_input_index].func.1;
        //                 if let Some(output_type) = &self.api_functions[offer_function_index].output{
        //                     println!("output type: {:?}", output_type);
        //                     let input_param = &self.api_functions[*i1].inputs[input_index1];
        //                     let call_type = api_util::_same_type(&output_type, &input_param, true, &self.full_name_map);
        //                     match &call_type {
        //                         CallType::_NotCompatible => {
        //                             println!("Something wrong in connect unsafe");
        //                         }
        //                         _ => {
        //                             merged_seq.functions[merged_seq_len-1].params[input_index1] = 
        //                                 (
        //                                     ParamType::_FunctionReturn,
        //                                     offer_input_index,
        //                                     call_type.clone(),
        //                                 );
        //                             // merge sequence 可能会重复
        //                             // TODO:(选取fuzzable params最少的那个)
        //                             println!("merged seq: {}", merged_seq);
        //                             if !self.record_sequences.contains(&merged_seq.get_sequence_string()){
        //                                 // println!("merged seq1: {}", merged_seq);
        //                                 // println!("{:#?}", merged_seq);
        //                                 self.insert_unvisited_api_seq(&unvisited_sequences_map, &merged_seq, offer_input_index, seq2_len-1, merged_seq_len-1, &ty);
        //                                 self.record_sequences.push(merged_seq.get_sequence_string().clone());
        //                             }
        //                         }
        //                     }
                            
        //                     // println!("Merged seq: {}", merged_seq);
        //                     // self.api_sequences.push(merged_seq);
        //                 }
        //             }
        //         }
        //     }
        // }
    }

    pub fn reverse_collect_unsafe_sequence (&mut self, unsafe_index: usize) -> Option<Vec<ApiSequence>>{
        let result = self.reverse_search_sequence(unsafe_index, 0);
        result
    }

    pub fn reverse_search_sequence(&mut self, index: usize, len: usize) -> Option<Vec<ApiSequence>> {
        // 判断是不是开始方法，如果是，直接结束
        // 临时采取的办法：用len记录长度，当长度大于4时停止，避免无限循环。
        if len > 4 || self.stop_list.contains(&index) { return None; }
        if self.api_generation_map.contains_key(&index) {
            return Some(self.api_generation_map[&index].clone());
        }
        let function = &self.api_functions[index].clone();
        let mut sequences = Vec::new();
        let mut sequence = ApiSequence::new();
        let mut api_call = ApiCall::new(index, function.output.clone(), function._is_generic_function());
        match &function.output {
            Some(ty) => {
                if !api_util::_is_end_type(&ty, &self.full_name_map) {
                    api_call.set_output_type(Some(ty.clone()));
                }
            },
            None => { }
        }

        if !function.unsafe_info.is_empty(){
            api_call.is_unsafe = true;
        }
        if !function.rawptr_info.is_empty(){
            api_call.is_get_raw_ptr = true;
        }
        if !function.drop_info.is_empty(){
            api_call.is_drop = true;
        }
        if !function.mutate_info.is_empty(){
            api_call.is_mutate = true;
        }
        // 如果是个unsafe函数，给sequence添加unsafe标记
        if function._unsafe_tag._is_unsafe() {
            sequence.set_unsafe();
        }
        if function._trait_full_path.is_some() {
            let trait_full_path = function._trait_full_path.as_ref().unwrap();
            sequence.add_trait(trait_full_path);
        }
        let input_params = &function.inputs;
        let input_params_num = input_params.len();
        // let mut need_functions_index = Vec::new();

        if input_params_num == 0 {
            // 无需输入参数，直接可以满足
            // return api_call.clone();
            sequence._add_fn(api_call.clone());
            if !sequences.contains(&sequence){
                sequences.push(sequence);
            }
            // sequences.push(sequence);
            return Some(sequences);
        }
        // let tmp_sequence = sequence.clone();
        if !sequences.contains(&sequence){
            sequences.push(sequence.clone());
        }
        // sequences.push(sequence.clone());
        let mut call_fuzzable_nums = 0;
        let mut fuzzable_types = Vec::new();
        let mut api_call_map: HashMap<String, ApiCall> = HashMap::new();
        api_call_map.insert(sequence.get_sequence_string(), api_call.clone());
        for i in 0..input_params_num {
            let input_ty = &input_params[i];
            if api_util::is_fuzzable_type(input_ty, &self.full_name_map){
                // 如果参数是fuzzable的
                let current_fuzzable_index = call_fuzzable_nums;
                let fuzzable_call_type =
                    fuzzable_type::fuzzable_call_type_by_clean_type(input_ty, &self.full_name_map);
                let (fuzzable_type, call_type) =
                    fuzzable_call_type.generate_fuzzable_type_and_call_type();

                //如果出现了下面这段话，说明出现了Fuzzable参数但不知道如何参数化的
                //典型例子是tuple里面出现了引用（&usize），这种情况不再去寻找dependency，直接返回无法添加即可
                match &fuzzable_type {
                    FuzzableType::NoFuzzable => {
                        println!("Fuzzable Type Error Occurs!");
                        println!("type = {:?}", input_ty);
                        println!("fuzzable_call_type = {:?}", fuzzable_call_type);
                        println!("fuzzable_type = {:?}", fuzzable_type);
                        break;
                    }
                    _ => {}
                }

                //判断要不要加mut tag
                if api_util::_need_mut_tag(&call_type) {
                    sequence._insert_fuzzable_mut_tag(current_fuzzable_index);
                }

                //添加到sequence中去
                // sequence.fuzzable_params.push(fuzzable_type);
                fuzzable_types.push(fuzzable_type);
                for seq in &sequences {
                    let mut api_call_ = api_call_map.get(&seq.get_sequence_string()).unwrap().clone();
                    api_call_._add_param(
                        ParamType::_FuzzableType,
                        current_fuzzable_index,
                        call_type.clone(),
                    );
                    api_call_map.insert(seq.get_sequence_string(), api_call_);
                }
                call_fuzzable_nums += 1;
                // sequence._add_fn(api_call.clone());
                // sequences.push(sequence.clone());
                continue;
                // return Some(sequence_);
            }
            
            // 要考虑input_ty和可生成的type的转换 比如T可生成，那么Option<T>也可生成
            if let Some(reachable_dependencies) = &self.get_reachable_dependencies(index, i) {
                // 考虑是第几个input的
                let tmp_sequences = sequences.clone();
                sequences.clear();
                // 对于每个可以满足这个的方法，都去试一下:
                for dependency in reachable_dependencies {
                    let (api_type, function_index) = &dependency.input_fun;
                    let call_type = &dependency.call_type;
                    // 判断函数是否等于本身,先不加入自身
                    // 或者该function之前已经判断不可达
                    if *function_index == index || self.stop_list.contains(function_index) {
                        continue;
                    }
                    // 回溯递归获得序列，如果成功，则说明该序列是可达的
                    if let Some(mut pre_sequences) = self.reverse_search_sequence(*function_index, len+1) {
                        for pre_sequence in &mut pre_sequences {
                            let need_index = pre_sequence.functions.len() - 1;
                            //参数需要加mut 标记的话
                            if api_util::_need_mut_tag(&call_type) {
                                pre_sequence._insert_function_mut_tag(need_index);
                            }
                            //如果call type是unsafe的，那么给sequence加上unsafe标记
                            // if call_type.unsafe_call_type()._is_unsafe() {
                            //     pre_sequence.set_unsafe();
                            // }
                            // pre_sequence._add_fn(api_call.clone());
                            for s in &tmp_sequences {
                                let mut api_call_ = api_call_map.get(&s.get_sequence_string()).unwrap().clone();
                                // let mut api_call_ = api_call.clone();
                                api_call_._add_param(
                                    ParamType::_FunctionReturn,
                                    need_index + s.functions.len(),
                                    call_type.clone(),
                                );
                                let new_sequence = s._merge_another_sequence(pre_sequence);
                                api_call_map.insert(new_sequence.get_sequence_string(), api_call_);
                                if !sequences.contains(&new_sequence) {
                                    sequences.push(new_sequence);
                                }
                                // sequences.push(new_sequence.clone());
                            }
                        }
                    }
                }
            } else {
                if !self.stop_list.contains(&index) {
                    self.stop_list.push(index);
                }
                return None;
            }
        }
        // 直到这里，序列才完整。对于每个序列，插入该方法的api_call
        for seq in &mut sequences {
            // 更新fuzzable params的指向关系
            let fuzz_len = seq.fuzzable_params.len();
            for fuzzable_type in &fuzzable_types{
                seq.fuzzable_params.push(fuzzable_type.clone());
            }
            let mut api_call_ = api_call_map.get(&seq.get_sequence_string()).unwrap().clone();
            for param in &mut api_call_.params {
                match param.0 {
                    ParamType::_FunctionReturn => {},
                    ParamType::_FuzzableType => {
                        param.1 += fuzz_len;
                    }
                }
            }
            seq._add_fn(api_call_);
        }
        if !self.api_generation_map.contains_key(&index) {
            self.api_generation_map.insert(index, sequences.clone());
        }
        return Some(sequences)
    }

    // 获取到达目标Parameter的序列
    pub fn get_sequence_for_parameter(&mut self, target_param: &ApiParameter) -> Option<Vec<ApiSequence>> {
        let mut res_sequences: Vec<ApiSequence> = Vec::new();
        // let target_param = &self.api_parameters[param_index];
        for func_index in target_param.get_return_functions() {
            if let Some(sequences) = self.reverse_search_sequence(func_index, 0) {
                for seq in &sequences {
                    if !res_sequences.contains(seq) {
                        res_sequences.push(seq.clone());
                    }
                }
            }
        }
        if res_sequences.len() > 0 {
            return Some(res_sequences);
        } else {
            return None;
        }
    }

    // 通过预设模式来寻找api序列
    // 以Paramater为key
    pub fn pattern_based_search(&mut self) {
        self.api_sequences.clear();

        // 进行模式匹配：
        // 先判断struct是否满足（即每个模式都有对应的API）
        // 按顺序生成序列
        println!(">>>>>>>>>> PATTERN BASED SEARCH <<<<<<<<<<");
        println!("Current path:{:?}", std::env::current_dir());
        let mut config_path = PathBuf::from("/home/wubohao/Lab/Rust-Lib-Testing/");
        config_path.push("pattern.txt");
        println!("pattern config file: {:?}", config_path);
        let config_file = File::open(config_path).unwrap();
        let config_reader = BufReader::new(config_file);
        for line in config_reader.lines() {
            match line {
                Ok(pattern) => {
                    if pattern.starts_with("//") | pattern.starts_with("#") {
                        continue;
                    }
                    println!("\n>>>>>>>>> PATTERN {:?} START", pattern);
                    let operations: Vec<usize> = pattern.split(" ").map(
                        |op| op.parse::<usize>().unwrap_or(0)
                    ).collect();
                    let mut pattern_sequences_map: HashMap<String, Vec<ApiSequence>> = HashMap::new();
                    let mut current_pattern = String::new();
                    for operate in &operations {
                        print!("\n>>>>>>> OPERATION {:?} ", operate);
                        pattern_sequences_map = match operate {
                            1 => self.pattern_seq_generator(&pattern_sequences_map, PatternOperation::Unsafe),
                            2 => self.pattern_seq_generator(&pattern_sequences_map, PatternOperation::GetRawPointer),
                            3 => self.pattern_seq_generator(&pattern_sequences_map, PatternOperation::Drop),
                            4 => self.pattern_seq_generator(&pattern_sequences_map, PatternOperation::Use),
                            5 => self.pattern_seq_generator(&pattern_sequences_map, PatternOperation::Mutate),
                            _ => { println!("WRONG OPERATION FOUND"); break;}
                        };
                        if pattern_sequences_map.is_empty() {
                            // 即无法满足pattern
                            println!("PATTERN {} FAILED <<<<<<<<<\n", current_pattern);
                            break;
                        } else {
                            // println!("sequences_map: {:#?}", pattern_sequences_map);
                            current_pattern += &operate.to_string();
                            for (type_name, pattern_seqs) in &pattern_sequences_map {
                                for pattern_seq in pattern_seqs {
                                    println!("{}", pattern_seq.get_sequence_string());
                                }
                            }
                            println!("CURRENT PATTERN {} SUCCESS <<<<<<<\n", current_pattern);
                        }
                    }
                    for (type_name, pattern_seqs) in &pattern_sequences_map{
                        for pattern_seq in pattern_seqs {
                            let mut finished_pattern_seq = pattern_seq.clone();
                            finished_pattern_seq._pattern_mark = Some(pattern.clone());
                            println!("SEQUENCE: {}", finished_pattern_seq);
                            self.api_sequences.push(finished_pattern_seq);
                        }
                    }
                    if !pattern_sequences_map.is_empty() {
                        println!("PATTERN {} SUCCESS <<<<<<<<<\n", current_pattern);
                    }
                },
                _ => {
                    println!("ERROR FOUND: {:?}", line);
                }
            }
        }
        println!(">>>>>>>>>> PATTERN BASED SEARCH <<<<<<<<<<");
    }

    // seq_map: 数据类型 -> 对应的api序列
    pub fn pattern_seq_generator(
        &mut self, 
        seq_map: &HashMap<String, Vec<ApiSequence>>, 
        operation: PatternOperation
    ) -> HashMap<String, Vec<ApiSequence>> {
        let new_seq_map = match operation {
            PatternOperation::Unsafe | PatternOperation::GetRawPointer => {
                self.pattern_seq_generator_helper1(seq_map, operation)
            },
            PatternOperation::Drop => {
                // 第一种情况，没有返回值导致的Drop
                self.pattern_seq_generator_helper1(seq_map, operation)
                // 第二种情况，有返回值，所以需要制造drop
            },
            PatternOperation::Use => {
                self.pattern_seq_generator_helper2(seq_map, operation)
            },
            PatternOperation::Mutate => {
                self.pattern_seq_generator_helper1(seq_map, operation)
            }
        };
        new_seq_map
    }

    pub fn pattern_seq_generator_helper1(
        &mut self, 
        seq_map: &HashMap<String, Vec<ApiSequence>>, 
        operation: PatternOperation
    ) -> HashMap<String, Vec<ApiSequence>> {
        let mut res_seq_map: HashMap<String, Vec<ApiSequence>> = HashMap::new();
        for param_index in 0..self.api_parameters.len() {
            let param = &self.api_parameters[param_index].clone();
            for (func_index, param_index) in param.get_use_functions() {
                // println!("param: {} -> use function: {}-{}", param.as_string(), func_index, param_index);
                let func = &self.api_functions[func_index];
                let mut moved_drop_flag = false; // 通过不使用返回值来drop
                let mut flag: bool = match operation {
                    PatternOperation::Unsafe => {
                        let mut flag_ = false;
                        if !func.unsafe_info.is_empty() | func._unsafe_tag._is_unsafe() { flag_ = true; }
                        flag_
                    },
                    PatternOperation::GetRawPointer => {
                        let mut flag_ = false;
                        if !func.rawptr_info.is_empty() { flag_ = true; }
                        flag_
                    },
                    PatternOperation::Drop => {
                        let mut flag_ = false;
                        if !func.drop_info.is_empty() { flag_ = true; }
                        flag_
                    },
                    PatternOperation::Mutate => {
                        let mut flag_ = false;
                        if !func.mutate_info.is_empty() { flag_ = true; }
                        flag_
                    }
                    _ => {
                        false
                    }
                };
                // 返回值不接受导致drop的情况
                if !flag && operation == PatternOperation::Drop {
                    match &func.output {
                        Some(output_type) => {
                            if !output_type.is_primitive() {
                                flag = true;
                                moved_drop_flag = true;
                            }
                        }
                        None => {}
                    }
                }

                if flag {
                    println!("\nparam: {} -> use function: {}-{}", param.as_string(), func_index, param_index);
                    let param_string = param.as_string();
                    if seq_map.is_empty() {
                        let mut new_seq = ApiSequence::new();
                        // mark seq if function-parm is generic
                        if let Some(generic_path_name) = &self.api_functions[func_index].get_generic_path_by_param_index(param_index) {
                            // GENERIC TYPE
                            println!("EMPTY GENERIC TYPE");
                            if !param.is_generic(){
                                new_seq.add_generic_info(&generic_path_name, param.clone());
                            } else {
                                continue;
                            }
                        }
                        if let Some(mut new_seqs) = self.try_to_call_func(&new_seq, func_index) {
                            if moved_drop_flag {
                                for seq_ in &mut new_seqs {
                                    seq_._insert_move_index(seq_.functions.len() - 1);
                                }
                            }
                            res_seq_map.insert(param_string.clone(), new_seqs);
                        }
                    } else if let Some(current_seqs) = seq_map.get(&param_string) {
                        for current_seq in current_seqs {
                            let mut currenet_seq_ = current_seq.clone();
                            if let Some(generic_path_name) = &self.api_functions[func_index].get_generic_path_by_param_index(param_index) {
                                if let Some(generic_parameter) = currenet_seq_.get_generic_parameter(&generic_path_name) {
                                    // 有了要看是不是一样的
                                    if param.as_string() != generic_parameter.as_string() {
                                        println!("DIFFERENT GENERIC TYPE MAP");
                                        continue;
                                    }
                                } else {
                                    println!("GENERIC TYPE");
                                    if !param.is_generic() {
                                        currenet_seq_.add_generic_info(&generic_path_name, param.clone());
                                    } else {
                                        continue;
                                    }
                                }
                            }
                            if let Some(new_seqs) = self.try_to_call_func(&currenet_seq_, func_index) {
                                if let Some(res_seqs) = res_seq_map.get_mut(&param_string) {
                                    for new_seq in new_seqs {
                                        if !res_seqs.contains(&new_seq) {
                                            let mut new_seq_ = new_seq.clone();
                                            if moved_drop_flag {
                                                new_seq_._insert_move_index(new_seq.functions.len() - 1);
                                            }
                                            res_seqs.push(new_seq_.clone());
                                        }
                                    }
                                } else {
                                    res_seq_map.insert(param_string.clone(), new_seqs);
                                }
                            // println!("opeartion seqs: {:?}", operation_seqs);
                            }
                        }
                    }
                    println!("after call function");
                    for (api_param_name, seqs_) in &res_seq_map {
                        if api_param_name == &param_string {
                            for seq_ in seqs_ {
                                println!("{}", seq_.get_sequence_string());
                            }
                        }
                    }
                } else {
                    println!("param is not satisified");
                }
            }
        }
        res_seq_map
    }

    pub fn pattern_seq_generator_helper2(
        &mut self, 
        seq_map: &HashMap<String, Vec<ApiSequence>>, 
        operation: PatternOperation
    ) -> HashMap<String, Vec<ApiSequence>> {
        let mut res_seq_map: HashMap<String, Vec<ApiSequence>> = HashMap::new();
        // 分析需要对哪些变量能use
        for (type_, seqs) in seq_map {
            let mut new_seqs = Vec::new();
            for seq in seqs {
                if let Some(usable_params) = self.return_usable_analyse(&seq) {
                    let mut new_seq = seq.clone();
                    for i in 1..usable_params.len() + 1 {
                        let index = usable_params.len() - i;
                        let mut api_call = ApiCall::new_without_params(&ApiType::UseParam, 0);
                        // TODO: 分析CallType是Display还是Debug还是Call其它方法
                        if usable_params[index] {
                            println!("using index {}-{}", seq, index);
                            let func_index = new_seq.functions[index].func.1;
                            let usable_function = &self.api_functions[func_index];
                            let output_type = usable_function.output.as_ref().unwrap();
                    
                            let ouput_inner_type = api_parameter::get_inner_type(output_type);
                            let parameter = match self.find_api_parameter_by_clean_type(&ouput_inner_type) {
                                Some(parameter_) => parameter_,
                                None => {
                                    // check 是否 generic 情况
                                    if api_parameter::is_generic_type(output_type) {
                                        // 需要根据sequence里的generic_map替换类型
                                        if let Some(generic_path) = usable_function.get_generic_path_for_output() {
                                            if let Some(parameter_) = seq.get_generic_parameter(&generic_path) {
                                                parameter_
                                            } else {
                                                continue;
                                            }
                                        } else {
                                            continue;
                                        }
                                    } else {
                                        // 如果generic也找不到就跳过了
                                        println!("Parameter for {} can't found", index);
                                        continue;
                                    }
                                }
                            };
                            if parameter.is_implement_debug_trait() {
                                println!("Parameter-{} implement debug trait", parameter.as_string());
                                api_call._add_param(
                                    ParamType::_FuzzableType,
                                    index,
                                    CallType::_Debug,
                                );
                                new_seq._add_fn(api_call);
                            } else if parameter.is_implement_display_trait() {
                                println!("Parameter-{} implement display trait", parameter.as_string());
                                api_call._add_param(
                                    ParamType::_FuzzableType,
                                    index,
                                    CallType::_Display,
                                );
                                let mut new_seq = seq.clone();
                                new_seq._add_fn(api_call);
                            } else {
                                println!("Parameter-{} can't print", parameter.as_string());
                                // 查看哪个funciton immutable调用
                                for (func_index, param_index) in &parameter.get_use_functions() {
                                    if let Some(use_seqs) = self.try_to_call_func(&new_seq, *func_index) {
                                        println!("call {}", func_index);
                                        new_seq = use_seqs[0].clone();
                                        // 最后一个api_call的使用的是index的param
                                        // let mut last_api_call = use_seq.functions[use_seq.len() - 1];
                                        // *last_api_call.func.0 = ApiType::UseParam;
                                        break;
                                    } else {
                                        println!("can't call {}", func_index);
                                    }
                                }
                            }
                        }
                    }
                    new_seqs.push(new_seq);
                }
            }
            res_seq_map.insert(type_.to_string(), new_seqs);
        }
        res_seq_map
    }

    // 尝试通过已有序列拓展到目标API
    pub fn try_to_call_func(&mut self, origin_sequence: &ApiSequence, func_id: usize) -> Option<Vec<ApiSequence>> {
        let mut result_sequences = Vec::new();
        let mut seq_api_call_map: HashMap<String, ApiCall> = HashMap::new();
        let sequence_len = origin_sequence.functions.len();
        // 查看目标API的input需求
        // 分析API的input是否可以从已有序列生成，或通过fuzzable生成
        // 若不能通过已有序列生成，则针对该数据类型获取生成序列，进行插入
        let input_function = &self.api_functions[func_id].clone();
        let api_type = match input_function._is_generic_function() {
            true => { ApiType::GenericFunction }
            false => { ApiType::BareFunction }
        };
        println!("\n>>>>> {} TRY TO CALL {}", origin_sequence, func_id);
        println!("1. TRY CALL FUNCTION DIRECTLY WITH THE ORIGIN SEQUENCE");
        if let Some(result_seq) = self.sequence_add_fun(&api_type, func_id, &origin_sequence) {
            // 可以直接添加该API
            println!("DIRECT CALL SUCCESS <<<<<\n");
            result_sequences.push(result_seq);
            return Some(result_sequences);
        }
        // println!("can not direct call the function using the origin sequence");
        let mut api_call = ApiCall::new(func_id, input_function.output.clone(), input_function._is_generic_function());
        if !input_function.unsafe_info.is_empty(){
            api_call.is_unsafe = true;
        }
        if !input_function.rawptr_info.is_empty(){
            api_call.is_get_raw_ptr = true;
        }
        if !input_function.drop_info.is_empty(){
            api_call.is_drop = true;
        }
        if !input_function.mutate_info.is_empty(){
            api_call.is_mutate = true;
        }
        result_sequences.push(origin_sequence.clone());
        // 存在不可到达的变量
        println!("2. TRY TO CALL FUNCTION AS MUCH AS POSSIBLE");
        for i in 0..input_function.inputs.len() {
            println!("CALLING FUNCTION PARAM {}-{}", func_id, i);
            let input_type = &input_function.inputs[i];
            let mut fuzzable_flag = false;
            // 是否是fuzzable
            println!("i). TRY TO CALL FUNCTION PARAM WITH FUZZABLE PARAMETER");
            for new_seq in &mut result_sequences {
                let fuzzable_call_type = match api_util::is_fuzzable_type(input_type, &self.full_name_map) {
                    true => fuzzable_type::fuzzable_call_type_by_clean_type(input_type, &self.full_name_map),
                    false => {
                        // 可能是generic，但可以用preluded type替代
                        let fuzz_call_type = fuzzable_type::fuzzable_call_type_by_clean_type(input_type, &self.full_name_map);
                        let mut final_call_type = FuzzableCallType::NoFuzzable;
                        // 需要替换FuzzableCallType::Generic为特定的类型。
                        // println!("origin fuzzable call type: {:#?}", fuzz_call_type);
                        if let Some(generic_path_name) = input_function.get_generic_path_by_param_index(i) {
                            if let Some(generic_parameter) = new_seq.get_generic_parameter(&generic_path_name) {
                                final_call_type = fuzzable_type::fuzzable_call_type_by_api_parameter(&generic_parameter, fuzz_call_type);
                            }
                        }
                        final_call_type
                    }
                };
                // println!("final fuzzable call type: {:#?}", fuzzable_call_type);
                let current_fuzzable_index = new_seq.fuzzable_params.len();
                let (fuzzable_type, call_type) = fuzzable_call_type.generate_fuzzable_type_and_call_type();
                if fuzzable_type == FuzzableType::NoFuzzable {
                    if fuzzable_flag {
                        panic!("fuzzable flag panic");
                    }
                    break;
                } else {
                    // println!("fuzzable input type");
                    fuzzable_flag = true;
                }
                if api_util::_need_mut_tag(&call_type) {
                    new_seq._insert_fuzzable_mut_tag(current_fuzzable_index);
                }
                new_seq.fuzzable_params.push(fuzzable_type);
                let new_api_call = match seq_api_call_map.get_mut(&new_seq.get_sequence_string()) {
                    Some(api_call_) => api_call_,
                    None => &mut api_call,
                };
                new_api_call._add_param(
                    ParamType::_FuzzableType,
                    current_fuzzable_index,
                    call_type
                );
            }
            if fuzzable_flag {
                println!("PROVIDED BY FUZZABLE PARAMETER");
                continue;
            }
            // println!("not fuzzable");
            println!("ii). TRY TO CALL FUNCTION PARAM BY ORIGIN SEQUENCE");
            // 再看sequence是否可以提供
            let mut sequence_support_flag = false;      
            for j in 0..sequence_len {
                if origin_sequence._is_moved(j) {
                    println!("APICALL-{} is moved", j);
                    continue;
                }
                let seq_func = &origin_sequence.functions[j];
                let (call_api_type, call_func_index) = seq_func.func;
                if let Some(dependency_index) = self.check_dependency(&call_api_type, call_func_index, &api_type, func_id, i) {
                    // 序列中能作为该func的输入
                    // sequence_support_flag = true;
                    let output_function = &self.api_functions[call_func_index];
                    let output_type = &output_function.output.as_ref().unwrap();
                    let dependency_ = self.api_dependencies[dependency_index].0.clone();
                    let interest_parameters = &dependency_.parameters;
                    let (is_call_parameter_generic, is_called_paramter_generic) = (output_function._is_generic_output(), input_function._is_generic_input(i));
                    match (is_call_parameter_generic, is_called_paramter_generic) {
                        (false, false) => {
                            // Do Nothing
                            sequence_support_flag = true;
                        },
                        (true, true) => {
                            let mut output_generic_path_name = String::new();
                            let mut input_generic_path_name = String::new();
                            let output_generic_parmeter = match api_parameter::get_generic_name(output_type) {
                                Some(symbol) => {
                                    output_generic_path_name = output_function.get_generic_path_by_symbol(&symbol).unwrap();
                                    let parameter = origin_sequence.get_generic_parameter(&output_generic_path_name);
                                    parameter
                                }
                                None => None
                            };
                            let input_generic_parmeter = match api_parameter::get_generic_name(input_type) {
                                Some(symbol) => {
                                    input_generic_path_name = input_function.get_generic_path_by_symbol(&symbol).unwrap();
                                    let parameter = origin_sequence.get_generic_parameter(&input_generic_path_name);
                                    parameter
                                }
                                None => None
                            };

                            match (output_generic_parmeter, input_generic_parmeter) {
                                // ouput(S) -> input(S)
                                (Some(output_parameter), Some(input_parameter)) => {
                                    if output_parameter.as_string() != input_parameter.as_string(){
                                        continue;
                                    }
                                },
                                // ouput(S) -> input(T)
                                (Some(output_parameter), None) => {
                                    let mut dependency_flag = false;
                                    for param in interest_parameters {
                                        if param.as_string() == output_parameter.as_string() {
                                            dependency_flag = true;
                                            for new_sequence in &mut result_sequences {
                                                let new_api_call = match seq_api_call_map.get(&new_sequence.get_sequence_string()) {
                                                    Some(api_call_) => api_call_.clone(),
                                                    None => api_call.clone(),
                                                };
                                                new_sequence.add_generic_info(&input_generic_path_name, param.clone());
                                                seq_api_call_map.insert(new_sequence.get_sequence_string(), new_api_call.clone());
                                            }
                                            break;
                                        }
                                    }
                                    if !dependency_flag { continue; }
                                },
                                // ouput(T) -> input(S)
                                (None, Some(input_parameter)) => {
                                    let mut dependency_flag = false;
                                    for param in interest_parameters {
                                        if param.as_string() == input_parameter.as_string() {
                                            dependency_flag = true;
                                            for new_sequence in &mut result_sequences {
                                                let new_api_call = match seq_api_call_map.get(&new_sequence.get_sequence_string()) {
                                                    Some(api_call_) => api_call_.clone(),
                                                    None => api_call.clone(),
                                                };
                                                new_sequence.add_generic_info(&output_generic_path_name, param.clone());
                                                seq_api_call_map.insert(new_sequence.get_sequence_string(), new_api_call.clone());
                                            }
                                            break;
                                        }
                                    }
                                    if !dependency_flag { continue; }
                                },
                                // ouput(T) -> input(T)
                                (None, None) => {
                                    // 在interestParameter选一个Parameter
                                    // 暂时采用随机选取
                                    // TODO: Parameter评分，基于sequence还是function
                                    let mut rng = rand::thread_rng();
                                    let random_num = rng.gen_range(0, interest_parameters.len());
                                    let param = &interest_parameters[random_num];
                                    for new_sequence in &mut result_sequences {
                                        let new_api_call = match seq_api_call_map.get(&new_sequence.get_sequence_string()) {
                                            Some(api_call_) => api_call_.clone(),
                                            None => api_call.clone(),
                                        };
                                        new_sequence.add_generic_info(&output_generic_path_name, param.clone());
                                        new_sequence.add_generic_info(&input_generic_path_name, param.clone());
                                        seq_api_call_map.insert(new_sequence.get_sequence_string(), new_api_call.clone());
                                    }
                                },
                            }
                            sequence_support_flag = true;
                        },
                        (false, true) => {
                            // SpecialType -> GenericType
                            if let Some(symbol) = api_parameter::get_generic_name(input_type) {
                                assert_eq!(interest_parameters.len(), 1);
                                let generic_path_name = input_function.get_generic_path_by_symbol(&symbol).unwrap();
                                if let Some(generic_parameter) = origin_sequence.get_generic_parameter(&generic_path_name) {
                                    // seq里有泛型的具体类型
                                    // 则只有一个选择，看能不能满足input
                                    if interest_parameters[0].as_string() != generic_parameter.as_string() {
                                        continue;
                                    }
                                } else {
                                    // 没有则添加具体类型
                                    for new_sequence in &mut result_sequences {
                                        let new_api_call = match seq_api_call_map.get(&new_sequence.get_sequence_string()) {
                                            Some(api_call_) => api_call_.clone(),
                                            None => api_call.clone(),
                                        };
                                        new_sequence.add_generic_info(&generic_path_name, interest_parameters[0].clone());
                                        seq_api_call_map.insert(new_sequence.get_sequence_string(), new_api_call.clone());
                                    }
                                }
                                sequence_support_flag = true;
                            } else { }
                        },
                        (true, false) => {
                            // GenericType -> SpecialType
                            // 确保GenericType和SpecialType的关系存在
                            if let Some(symbol) = api_parameter::get_generic_name(output_type) {
                                assert_eq!(interest_parameters.len(), 1);
                                let generic_path_name = output_function.get_generic_path_by_symbol(&symbol).unwrap();
                                if let Some(generic_parameter) = origin_sequence.get_generic_parameter(&generic_path_name) {
                                    // seq里有泛型的具体类型
                                    // 则只有一个选择，看能不能满足input
                                    if interest_parameters[0].as_string() != generic_parameter.as_string() {
                                        continue;
                                    }
                                } else {
                                    // 没有则添加具体类型
                                    // TODO: 如果上一个方法的返回值没有确定类型，能不能加函数？
                                    for new_sequence in &mut result_sequences {
                                        let new_api_call = match seq_api_call_map.get(&new_sequence.get_sequence_string()) {
                                            Some(api_call_) => api_call_.clone(),
                                            None => api_call.clone(),
                                        };
                                        new_sequence.add_generic_info(&generic_path_name, interest_parameters[0].clone());
                                        seq_api_call_map.insert(new_sequence.get_sequence_string(), new_api_call.clone());
                                    }
                                }
                                sequence_support_flag = true;
                            } else { }
                            
                        },
                    }
                    if sequence_support_flag {
                        sequence_support_flag = false;
                        let mut new_result_sequences = Vec::new();
                        for new_seq in &mut result_sequences { 
                            let mut new_api_call = match seq_api_call_map.get(&new_seq.get_sequence_string()) {
                                Some(api_call_) => api_call_.clone(),
                                None => api_call.clone(),
                            };
                            let dependency_ = self.api_dependencies[dependency_index].0.clone();
                            if api_util::_need_mut_tag(&dependency_.call_type) {
                                new_seq._insert_function_mut_tag(i);
                            }
                            new_api_call._add_param(
                                ParamType::_FunctionReturn,
                                j,
                                dependency_.call_type,
                            );
                            let mut test_seq = new_seq.clone();
                            test_seq._add_fn(new_api_call.clone());
                            match self.sequence_syntax_analyse(&test_seq) {
                                true => {
                                    // 有合法的才变true，方便后面判断要不要继续
                                    // 全部都不合法就是false继续找
                                    seq_api_call_map.insert(new_seq.get_sequence_string(), new_api_call.clone());
                                    // println!("seq support func: {}-{:#?}", new_seq.get_sequence_string(), new_api_call);
                                    new_result_sequences.push(new_seq.clone()); 
                                    sequence_support_flag = true;
                                },
                                false => { 
                                    println!("ANALYSE FAILED SEQ: {}", test_seq); 
                                }
                            }
                        }
                        if sequence_support_flag {
                            result_sequences = new_result_sequences;
                            break;
                        }
                    }
                }
            }
            if sequence_support_flag {
                println!("PROVIDED BY FUZZABLE PARAMETER");
                continue;
            }
            // 在type generation map中查找
            // 否则就调用函数生成
            
            // TODO: 这里逻辑需要再考虑一下
            // 可能本来就是generic_type
        
            // 该变量类型可以生成
            // 则合并seq和type generation sequence
            println!("iii). TRY TO CALL FUNCTION PARAM BY OTHER SEQUENCE");
            let mut other_support_flag = false;
            let mut new_result_sequences = Vec::new();
            for new_seq in &mut result_sequences {
                let mut parameter_string = String::new();
                let mut type_generation_sequences = Vec::new();
                match self.find_api_parameter_by_clean_type(&input_type) {
                    Some(param) => {
                        if let Some(sequences) = self.type_generation_map.get(&param.as_string()) {
                            type_generation_sequences = sequences.to_vec();
                            parameter_string = param.as_string();
                        } else {
                            match self.get_sequence_for_parameter(&param) {
                                Some(sequences) => {
                                    type_generation_sequences = sequences;
                                    parameter_string = param.as_string();
                                },
                                None => {continue;}
                            }
                        }
                    },
                    None => {
                        // generic, 则去api_parameter里面找是否有符合的
                        if api_parameter::is_generic_type(input_type) {
                            // check generic map
                            if let Some(generic_path_name) = input_function.get_generic_path_by_param_index(i) {
                                if let Some(api_param) = new_seq.get_generic_parameter(&generic_path_name) {
                                    // println!("MATCHED PARAMETER: {}", api_param.as_string());
                                    if let Some(sequences) = self.type_generation_map.get(&api_param.as_string()) {
                                        type_generation_sequences = sequences.to_vec();
                                        parameter_string = api_param.as_string();
                                    } else {
                                        match &self.get_sequence_for_parameter(&api_param) {
                                            Some(sequences) => {
                                                type_generation_sequences = sequences.to_vec();
                                                parameter_string = api_param.as_string();
                                            },
                                            None => {continue;}
                                        }
                                    }
                                } else {
                                    for j in 0..self.api_parameters.len() {
                                        let mut break_flag = false;
                                        for (func_index, param_index) in &self.api_parameters[j].get_use_functions() {
                                            if (func_index, param_index) == (&func_id, &i) && !self.api_parameters[j].is_generic() {
                                                // 该Parameter能作为该input
                                                // 该Parameter不能是含泛型的
                                                println!("{}-{}", func_id, i);
                                                println!("MATCHED PARAMETER: {}", self.api_parameters[j].as_string());
                                                let param = self.api_parameters[j].clone();
                                                if let Some(sequences) = self.type_generation_map.get(&param.as_string()) {
                                                    // new_seq.add_generic_info(&generic_path_name, param.clone());
                                                    type_generation_sequences = sequences.to_vec();
                                                    parameter_string = param.as_string();
                                                    break_flag = true;
                                                    break;
                                                } else {
                                                    match self.get_sequence_for_parameter(&param) {
                                                        Some(sequences) => {
                                                            // 导致错误了
                                                            // new_seq.add_generic_info(&generic_path_name, param.clone());
                                                            type_generation_sequences = sequences;
                                                            parameter_string = param.as_string();
                                                            break_flag = true;
                                                            break;
                                                        },
                                                        None => {continue;}
                                                    }
                                                }
                                            }
                                        }
                                        if break_flag { 
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    },
                };
                // if let Some(chosen_param) = self.find_api_parameter_by_string(&parameter_string) {
                //     if let Some(generic_path) = input_function.get_generic_path_by_param_index(i) {
                //         if let Some(param) = new_seq.get_generic_parameter(&generic_path) {
                //             assert!(param.as_string() == chosen_param.as_string());
                //         } else {
                //             // new_seq.add_generic_info(&generic_path, chosen_param.clone());
                //         }
                //     }
                // }
                if let Some(chosen_param) = self.find_api_parameter_by_string(&parameter_string) {
                    if let Some(generic_path) = input_function.get_generic_path_by_param_index(i) {
                        println!("generic match: {}-{}-{}", chosen_param.as_string(), func_id, i);
                        for type_generation_seq in &mut type_generation_sequences {
                            type_generation_seq.add_generic_info(&generic_path, chosen_param.clone());
                        }
                    }
                }
                for type_generation_seq in &type_generation_sequences {
                    let (call_api_type, index) = type_generation_seq.get_last_api();
                    if let Some(dependency_index) = self.check_dependency(
                        &call_api_type, index, &api_type, func_id, i
                    ) {
                        other_support_flag = true;
                        let mut new_api_call = match seq_api_call_map.get(&new_seq.get_sequence_string()) {
                            Some(api_call_) => api_call_.clone(),
                            None => api_call.clone(),
                        };
                        let mut result_seq = new_seq._merge_another_sequence(type_generation_seq);
                        let result_seq_len = result_seq.len();
                        let dependency_ = self.api_dependencies[dependency_index].0.clone();
                        if api_util::_need_mut_tag(&dependency_.call_type) {
                            result_seq._insert_function_mut_tag(result_seq_len - 1);
                        }
                        new_api_call._add_param(
                            ParamType::_FunctionReturn,
                            result_seq_len - 1,
                            dependency_.call_type,
                        );
                        // result_seq._add_fn(new_api_call);
                        seq_api_call_map.insert(result_seq.get_sequence_string(), new_api_call.clone());
                        println!("TYPE GENERATION SEQUENCE: {:#?}", type_generation_seq);
                        println!("MERGE RESULT SEQUENCE: {:#?}", result_seq);
                        new_result_sequences.push(result_seq);
                    }
                }
            }
            if other_support_flag {
                result_sequences = new_result_sequences;
                println!("PROVIED BY OTHER SEQUENCE");
            } else {
                println!("CAN'T SUPPORT THE FUNCTION PARAMETER {}-{}", func_id, i);
                println!("TRY TO CALL FAILED <<<<<\n");
                return None;
            }
        }
        // 每个input都能满足了
        // 对result_sequence添加ApiCall
        // println!("seq api call map: {:#?}", seq_api_call_map);
        for result_seq in &mut result_sequences {
            if let Some(api_call_) = seq_api_call_map.get(&result_seq.get_sequence_string()){
                result_seq._add_fn(api_call_.clone());
            }
        }

        // println!("result sequence: {}", result_sequences.len());

        if result_sequences.len() != 0 {
            println!("TRY TO CALL SUCCESS <<<<<\n");
            return Some(result_sequences);
        }
        println!("CAN'T SUPPORT THE FUNCTION PARAMETER {}", func_id);
        println!("TRY TO CALL FAILED <<<<<\n");
        None
    }

    // 根据所需数据类型，生成能生产该数据类型的序列
    // 同时更新type_generation_map
    pub fn type_sequence_generate(&mut self, func_id: usize, type_index: usize) -> Option<Vec<ApiSequence>> {
        let target_type = &self.api_functions[func_id].inputs[type_index].clone();
        let string_type = match self.find_api_parameter_by_clean_type(&target_type) {
            Some(param) => param.as_string(),
            None => { return None; }
        };
        
        for i in 0..self.api_functions.len() {
            let output_func = &self.api_functions[i];

            if let Some(output_type) = &output_func.output{
                let call_type = api_util::_same_type(
                    &output_type,
                    &target_type,
                    true,
                    &self.full_name_map
                );
                match &call_type {
                    CallType::_NotCompatible => {
                        continue;
                    }, 
                    _ => {
                        if let Some(sequences) = self.reverse_search_sequence(i, 0) {
                            if let Some(generation_sequences) = self.type_generation_map.get_mut(&string_type) {
                                for seq in &sequences {
                                    if !generation_sequences.contains(seq) {
                                        generation_sequences.push(seq.clone());
                                    }
                                }
                            } else {
                                self.type_generation_map.insert(string_type.to_string(), sequences);
                            }
                        }
                    }
                }
            }
        }
        if let Some(result_seqs) = self.type_generation_map.get(&string_type) {
            return Some(result_seqs.to_vec());
        } else {
            return None;
        }
    }

    pub fn weighted_unsafe_bfs(&mut self) {
        //清空所有的序列
        self.api_sequences.clear();
        self.reset_visited();

        let api_function_num = self.api_functions.len();
        // 构建初始seq
        // 1. unsafe api generation seq
        // 2. unvisited start api seq
        for i in 0..self.api_functions.len() {
            let func = &self.api_functions[i];
            if func.is_unsafe_function() {
                println!("Unsafe API {}: {}", i, self.api_functions[i].full_name);
                if let Some(mut unsafe_generation_sequences) = self.reverse_collect_unsafe_sequence(i) {
                    for unsafe_generation_sequence in &mut unsafe_generation_sequences {
                        if self.check_syntax(unsafe_generation_sequence) {
                            println!("unsafe sequence: {}", unsafe_generation_sequence);
                            for f in &unsafe_generation_sequence.functions {
                                self.api_functions_visited[f.func.1] = true;
                            }
                            self.record_sequences.push(unsafe_generation_sequence.get_sequence_string());
                        }
                    }
                    self.api_sequences.append(&mut unsafe_generation_sequences);
                }
            }
        }
 
        for i in 0..self.api_functions.len() {
            let function = &self.api_functions[i];
            let sequence = ApiSequence::new();
            let api_type = match function._is_generic_function() {
                true => { ApiType::GenericFunction }
                false => { ApiType::BareFunction }
            };
            if !self.api_functions_visited[i] {
                if let Some(new_sequence) = self.sequence_add_fun(&api_type, i, &sequence) {
                    println!("start sequence: {}", new_sequence);
                    self.api_sequences.push(new_sequence.clone());
                    self.record_sequences.push(new_sequence.get_sequence_string());
                    self.api_functions_visited[i] = true;
                }
            }
        }

        let mut tmp_sequences = self.api_sequences.clone();
        println!("Initial Seq Number: {}", self.api_sequences.len());
        for seq in &self.api_sequences {
            println!("{}", seq);
        }

        // <usize, Vec<HashSet<usize>>> 目标api -> 不可达的api
        // 初始化
        // let mut refuse_api_map: <usize, Vec<HashSet<usize>>> = HashMap::new();
        for func_index in 0..api_function_num {
            let mut refuse_fun_indexes = Vec::new();
            let input_num = self.api_functions[func_index].inputs.len();
            for i in 0..input_num {
                refuse_fun_indexes.push(HashSet::new());
            }
            self.refuse_api_map.insert(func_index, refuse_fun_indexes);
        }

        for api_func_index in 0..api_function_num {
            //查看权重
            let api_func = &self.api_functions[api_func_index];
            println!("NO.{}: {}, {}", api_func_index, api_func.full_name, api_func.weight);
        }
        
        // 接下来开始从长度1一直到max_len遍历
        let mut depth: i32 = 0;
        loop {
            let mut lose_weight = 0;
            let mut tmp_sequences_ = Vec::new();
            for i in 0..tmp_sequences.len() {
                let sequence = &tmp_sequences[i];
                // println!("L{} TS.{}: {}", depth, i, sequence);
                //长度为len的序列，去匹配每一个函数，如果可以加入的话，就生成一个新的序列
                let api_set = sequence.get_api_set();
                for api_func_index in 0..api_function_num {
                    let mut success_flag = false;
                    //查看权重是否小于等于0
                    let api_func = &self.api_functions[api_func_index];
                    if api_func.weight <= 0 {
                        continue;
                    }
                    let api_type = match api_func._is_generic_function() {
                        true => { ApiType::GenericFunction }
                        false => { ApiType::BareFunction }
                    };
                    println!("seq: {} add fun: {}", sequence, api_func.full_name);
                    if let Some(new_sequence) =
                        self.sequence_add_fun(&api_type, api_func_index, sequence)
                    {
                        if !self.record_sequences.contains(&new_sequence.get_sequence_string()) {
                            tmp_sequences_.push(new_sequence.clone());
                            self.record_sequences.push(new_sequence.get_sequence_string());
                            self.api_functions_visited[api_func_index] = true;
                            success_flag = true;
                        }
                    } else {
                        // update speed up record
                    }
                    if success_flag {
                        self.api_functions[api_func_index].weight -= 1;
                        lose_weight += 1;
                    }
                }
            }
            
            tmp_sequences_.sort();
            tmp_sequences_.reverse();
            println!("BFS Deep: {}, Seq Number: {}, Weight: {}/{}", depth, tmp_sequences_.len(), lose_weight, self.total_weight);
            tmp_sequences = tmp_sequences_;

            self.api_sequences.append(&mut tmp_sequences.clone());
            if tmp_sequences.len() == 0{
                println!("Can not search more node, Depth:{}", depth);
                break;
            }
            depth += 1;
            println!("{}, {}",lose_weight as f32, (self.total_weight as f32) * (((depth + 1) as f32).log(5_f32) as f32));
            if lose_weight as f32 >= (self.total_weight as f32) * (((depth + 1) as f32).log(5_f32) as f32) {
                self.set_weight();
            }
        }
        self.api_sequences.sort();
        self.api_sequences.reverse();
        println!("Unsafe BFS Seq Number: {}", self.api_sequences.len());
    }

    pub fn unsafe_bfs(&mut self, max_len: usize, stop_at_end_function: bool, fast_mode: bool) {
        //清空所有的序列
        self.api_sequences.clear();
        self.reset_visited();
        if max_len < 1 {
            return;
        }

        let api_function_num = self.api_functions.len();
        // 构建初始seq
        // 1. unsafe api generation seq
        // 2. unvisited start api seq
        for i in 0..self.api_functions.len() {
            let func = &self.api_functions[i];
            if func.is_unsafe_function() {
                println!("Unsafe API {}: {}", i, self.api_functions[i].full_name);
                if let Some(mut unsafe_generation_sequences) = self.reverse_collect_unsafe_sequence(i) {
                    for unsafe_generation_sequence in &mut unsafe_generation_sequences{
                        println!("{}", unsafe_generation_sequence);
                        for f in &unsafe_generation_sequence.functions {
                            self.api_functions_visited[f.func.1] = true;
                        }
                        self.record_sequences.push(unsafe_generation_sequence.get_sequence_string());
                    }
                    self.api_sequences.append(&mut unsafe_generation_sequences);
                }
            }
        }
 
        for i in 0..self.api_functions.len() {
            let function = &self.api_functions[i];
            let sequence = ApiSequence::new();
            let api_type = match function._is_generic_function() {
                true => { ApiType::GenericFunction }
                false => { ApiType::BareFunction }
            };
            if !self.api_functions_visited[i] {
                if let Some(new_sequence) = self.sequence_add_fun(&api_type, i, &sequence) {
                    self.api_sequences.push(new_sequence.clone());
                    self.record_sequences.push(new_sequence.get_sequence_string());
                    self.api_functions_visited[i] = true;
                }
            }
        }

        let mut tmp_sequences = self.api_sequences.clone();
        println!("Initial Seq Number: {}", self.api_sequences.len());
        for seq in &self.api_sequences {
            println!("{}", seq);
        }

        // <usize, Vec<HashSet<usize>>> 目标api -> 不可达的api
        // 初始化
        // let mut refuse_api_map: <usize, Vec<HashSet<usize>>> = HashMap::new();
        for func_index in 0..api_function_num {
            let mut refuse_fun_indexes = Vec::new();
            let input_num = self.api_functions[func_index].inputs.len();
            for i in 0..input_num {
                refuse_fun_indexes.push(HashSet::new());
            }
            self.refuse_api_map.insert(func_index, refuse_fun_indexes);
        }
        
        // 接下来开始从长度1一直到max_len遍历
        for len in 0..max_len - 1 {
            let mut tmp_sequences_ = Vec::new();
            for i in 0..tmp_sequences.len() {
                let sequence = &tmp_sequences[i];
                // println!("L{} TS.{}: {}", len, i, sequence);
                //长度为len的序列，去匹配每一个函数，如果可以加入的话，就生成一个新的序列
                let api_set = sequence.get_api_set();
                for api_func_index in 0..api_function_num {
                    //访问过且为unconditional api则直接跳过
                    if self.is_uncondition_api(api_func_index) && self.api_functions_visited[api_func_index] {
                        continue;
                    }
                    let function = &self.api_functions[api_func_index];
                    let api_type = match function._is_generic_function() {
                        true => { ApiType::GenericFunction }
                        false => { ApiType::BareFunction }
                    };
                    
                    if let Some(new_sequence) =
                        self.sequence_add_fun(&api_type, api_func_index, sequence)
                    {
                        if !self.record_sequences.contains(&new_sequence.get_sequence_string()) {
                            tmp_sequences_.push(new_sequence.clone());
                            self.record_sequences.push(new_sequence.get_sequence_string());
                            // self.api_sequences.push(new_sequence);
                            self.api_functions_visited[api_func_index] = true;
                        }

                        //bfs fast，如果都已经别访问过，直接退出
                        if self.check_all_visited() {
                            //println!("bfs all visited");
                            //return;
                        }
                    } else {
                        // update speed up record
                    }
                }
            }
            // println!("Before BFS Deep: {}, Seq Number: {}", len, tmp_sequences.len());
            // for seq in &tmp_sequences_ {
            //     println!("{}", seq);
            // }
            tmp_sequences_.sort();
            tmp_sequences_.reverse();
            // println!("BFS Deep: {}, Before Screening Seq Number: {}", len, tmp_sequences_.len());
            // if len == 0 {
            //     tmp_sequences = self.screen_sequence2(&tmp_sequences_);
            // } else {ß
            tmp_sequences = tmp_sequences_;
            // }
            println!("BFS Deep: {}, After Screening Seq Number: {}", len, tmp_sequences.len());
            self.api_sequences.append(&mut tmp_sequences.clone());
            // println!("Refuse Map: {:#?}", self.refuse_api_map);
            // for seq in &tmp_sequences {
            //     println!("{}", seq);
            // }
            
            if tmp_sequences.len() == 0{
                println!("Can not search more node");
                break;
            }
        }
        self.api_sequences.sort();
        self.api_sequences.reverse();
        println!("Unsafe BFS Seq Number: {}", self.api_sequences.len());

        //println!("There are total {} sequences after bfs", self.api_sequences.len());
        if !stop_at_end_function {
            std::process::exit(0);
        }
    }

    // 用来筛选bfs过程中的序列，减少bfs基数
    // 利用启发式来筛选
    pub fn screen_sequence(&self, sequences: &Vec<ApiSequence>) -> Vec<ApiSequence>{
        // 挑选最好的几个序列
        // 只要包含新node; edge; 都加入
        // edge先不考虑
        let mut res_sequences = Vec::new();
        let mut chosen_indexes = HashSet::new();
        let mut total_covered_nodes = HashSet::new();
        let mut total_covered_edges = HashSet::new();
        if sequences.len() == 0 {
            return res_sequences;
        }
        loop {
            let mut chosen_index = 0;
            let mut chosen_covered_nodes = 0;
            let mut chosen_covered_edges = 0;
            let mut unsafe_chosen_covered_nodes = 0;
            let mut unsafe_chosen_covered_edges = 0;
            let mut chosen_sequence_len = 0;
            let mut last_unsafe_nodes = 0;
            for i in 0..sequences.len() {
                let seq = &sequences[i];
                let mut unsafe_nodes = HashSet::new();
                if chosen_indexes.contains(&i) {
                    continue;
                }

                // node覆盖情况
                let covered_nodes = seq._get_contained_api_functions();
                let unsafe_covered_nodes = seq._get_contained_unsafe_api_functions(self);
                let mut additional_covered_nodes = 0;
                let mut unsafe_additional_covered_nodes = 0;
                for covered_node in &covered_nodes {
                    if !total_covered_nodes.contains(covered_node) {
                        additional_covered_nodes += 1;
                    }
                }
                for unsafe_covered_node in &unsafe_covered_nodes {
                    if !total_covered_nodes.contains(unsafe_covered_node) {
                        unsafe_additional_covered_nodes += 1;
                    }
                    if !unsafe_nodes.contains(unsafe_covered_node) {
                        unsafe_nodes.insert(unsafe_covered_node);
                    }
                }

                // edge覆盖情况
                let covered_edges = &seq._covered_dependencies;
                let mut additional_covered_edges = 0;
                let mut unsafe_additional_covered_edges = 0;
                for edge in covered_edges {
                    if !total_covered_edges.contains(edge) {
                        additional_covered_edges += 1;
                        if edge.1 {
                            unsafe_additional_covered_edges += 1;
                        }
                    }
                }

                //Node 优先
                if (additional_covered_nodes > chosen_covered_nodes)
                    || (additional_covered_nodes == chosen_covered_nodes 
                        && unsafe_additional_covered_nodes > unsafe_chosen_covered_nodes)
                    || (additional_covered_nodes == chosen_covered_nodes 
                        && unsafe_additional_covered_nodes == unsafe_chosen_covered_nodes
                        && additional_covered_edges > chosen_covered_edges)
                    || (additional_covered_nodes == chosen_covered_nodes 
                        && unsafe_additional_covered_nodes == unsafe_chosen_covered_nodes
                        && additional_covered_edges == chosen_covered_edges
                        && unsafe_additional_covered_edges > unsafe_chosen_covered_edges)
                    || (additional_covered_nodes == chosen_covered_nodes 
                        && unsafe_additional_covered_nodes == unsafe_chosen_covered_nodes
                        && additional_covered_edges == chosen_covered_edges
                        && unsafe_additional_covered_edges == unsafe_chosen_covered_edges
                        && unsafe_nodes.len() > last_unsafe_nodes)
                    || (additional_covered_nodes == chosen_covered_nodes 
                        && unsafe_additional_covered_nodes == unsafe_chosen_covered_nodes
                        && additional_covered_edges == chosen_covered_edges
                        && unsafe_additional_covered_edges == unsafe_chosen_covered_edges
                        && unsafe_nodes.len() == last_unsafe_nodes
                        && seq.len() < chosen_sequence_len)
                {
                    chosen_index = i;
                    chosen_sequence_len = seq.len();
                    chosen_covered_nodes = additional_covered_nodes;
                    chosen_covered_edges = additional_covered_edges;
                    unsafe_chosen_covered_nodes = unsafe_additional_covered_nodes;
                    unsafe_chosen_covered_edges = unsafe_additional_covered_edges;
                    last_unsafe_nodes = seq.len();
                }

                // // Edge 优先
                // if (additional_covered_edges > chosen_covered_edges)
                //     || (additional_covered_edges == chosen_covered_edges 
                //         && unsafe_additional_covered_edges > unsafe_chosen_covered_edges)
                //     || (additional_covered_edges == chosen_covered_edges 
                //         && unsafe_additional_covered_edges == unsafe_chosen_covered_edges
                //         && additional_covered_nodes > chosen_covered_nodes)
                //     || (additional_covered_edges == chosen_covered_edges 
                //         && unsafe_additional_covered_edges == unsafe_chosen_covered_edges
                //         && additional_covered_nodes == chosen_covered_nodes
                //         && unsafe_additional_covered_nodes > unsafe_chosen_covered_nodes)
                //     || (additional_covered_nodes == chosen_covered_nodes 
                //         && unsafe_additional_covered_nodes == unsafe_chosen_covered_nodes
                //         && additional_covered_edges == chosen_covered_edges
                //         && unsafe_additional_covered_edges == unsafe_chosen_covered_edges
                //         && unsafe_nodes.len() > last_unsafe_nodes)
                //     || (additional_covered_nodes == chosen_covered_nodes 
                //         && unsafe_additional_covered_nodes == unsafe_chosen_covered_nodes
                //         && additional_covered_edges == chosen_covered_edges
                //         && unsafe_additional_covered_edges == unsafe_chosen_covered_edges
                //         && unsafe_nodes.len() == last_unsafe_nodes
                //         && seq.len() > chosen_sequence_len)
                // {
                //     chosen_index = i;
                //     chosen_sequence_len = seq.len();
                //     chosen_covered_nodes = additional_covered_nodes;
                //     chosen_covered_edges = additional_covered_edges;
                //     unsafe_chosen_covered_nodes = unsafe_additional_covered_nodes;
                //     unsafe_chosen_covered_edges = unsafe_additional_covered_edges;
                //     last_unsafe_nodes = seq.len();
                // }
            }
            let chosen_seq = &sequences[chosen_index];
            if chosen_covered_nodes + unsafe_chosen_covered_nodes <= 0 
                && chosen_covered_edges + unsafe_chosen_covered_edges<= 0 {
                break;
            }
            chosen_indexes.insert(chosen_index);

            // 更新covered node
            let chosen_nodes = chosen_seq._get_contained_api_functions();
            for node in chosen_nodes {
                total_covered_nodes.insert(node);
            }
            res_sequences.push(chosen_seq.clone());

            // 更新coverd edge
            let chosen_edges = &chosen_seq._covered_dependencies;
            for edge in chosen_edges {
                total_covered_edges.insert(edge);
            }

            if chosen_indexes.len() == sequences.len() {
                break;
            }
        }

        res_sequences
    }

    // 用来筛选bfs过程中的序列，减少bfs基数
    // 删除子序列
    pub fn screen_sequence2(&self, sequences: &Vec<ApiSequence>) -> Vec<ApiSequence> {
        let mut res_sequences = Vec::new();
        if sequences.len() == 0 {
            return res_sequences;
        }
        res_sequences.push(sequences[0].clone());
        for i in 0..sequences.len() {
            let seq = &sequences[i];
            let mut add_flag = true;
            // println!("SS.{}: {}", i, seq);
            for j in 0..res_sequences.len() {
                let res_seq = &res_sequences[j];
                if res_seq.is_sub_seq(&seq) {
                    add_flag = false;
                    break;
                }
            }
            if add_flag {
                res_sequences.push(seq.clone());
            }
        }
        res_sequences
    }

    //生成函数序列，且指定调用的参数
    //加入对fast mode的支持
    pub fn bfs(&mut self, max_len: usize, stop_at_end_function: bool, fast_mode: bool) {
        //清空所有的序列
        self.api_sequences.clear();
        self.reset_visited();
        if max_len < 1 {
            return;
        }

        let api_function_num = self.api_functions.len();

        //无需加入长度为1的，从空序列开始即可，加入一个长度为0的序列作为初始
        let api_sequence = ApiSequence::new();
        self.api_sequences.push(api_sequence);

        //接下来开始从长度1一直到max_len遍历
        for len in 0..max_len {
            let mut tmp_sequences = Vec::new();
            for sequence in &self.api_sequences {
                if stop_at_end_function && self.is_sequence_ended(sequence) {
                    //如果需要引入终止函数，并且当前序列的最后一个函数是终止函数，那么就不再继续添加
                    continue;
                }
                if sequence.len() == len {
                    tmp_sequences.push(sequence.clone());
                }
            }
            for sequence in &tmp_sequences {
                //长度为len的序列，去匹配每一个函数，如果可以加入的话，就生成一个新的序列
                for api_func_index in 0..api_function_num {
                    //bfs fast, 访问过的函数不再访问
                    if fast_mode && self.api_functions_visited[api_func_index] {
                        continue;
                    }
                    let function = &self.api_functions[api_func_index];
                    let api_type = match function._is_generic_function() {
                        true => { ApiType::GenericFunction }
                        false => { ApiType::BareFunction }
                    };
                    if let Some(new_sequence) =
                        self.sequence_add_fun(&api_type, api_func_index, sequence)
                    {
                        self.api_sequences.push(new_sequence);
                        self.api_functions_visited[api_func_index] = true;

                        //bfs fast，如果都已经别访问过，直接退出
                        if self.check_all_visited() {
                            //println!("bfs all visited");
                            //return;
                        }
                    }
                }
            }
        }

        //println!("There are total {} sequences after bfs", self.api_sequences.len());
        if !stop_at_end_function {
            std::process::exit(0);
        }
    }

    // author：张彬
    pub fn dfs(&mut self, max_len: usize, fast_mode: bool) {
        // 清空所有的序列
        self.api_sequences.clear();
        self.reset_visited();
        if max_len < 1 {
            return;
        }

        let api_function_num = self.api_functions.len();
        let mut tmp_sequences: Vec<ApiSequence> = Vec::new();
        let mut tmp_len: usize = 0;
        let mut sequence = ApiSequence::new();
        let mut no_more = false;
        let mut end_loc = Vec::new();
        end_loc.resize(max_len, -1);
        tmp_sequences.resize(max_len, sequence);
        loop {
            let mut is_last_one = false;
            let mut take_last_one = false;
            if no_more {
                break;
            }
            if tmp_len <= 0 {
                sequence = ApiSequence::new();
            }else {
                sequence = tmp_sequences[tmp_len - 1].clone();
            }
            println!("current_sequence: {:?}", sequence);
            println!("end_loc: {:?}", end_loc);
            for api_func_index in 0..api_function_num {
                println!("tmp_len: {:?}, api_func_index: {:?}", tmp_len, api_func_index);
                if api_func_index >= api_function_num - 1 {
                    is_last_one = true;
                }
                if tmp_len < max_len - 1 && end_loc[tmp_len] >= api_func_index.try_into().unwrap() {
                    continue;
                }
                // if self.api_functions_visited[api_func_index] {
                //     continue;
                // }
                let function = &self.api_functions[api_func_index];
                let api_type = match function._is_generic_function() {
                    true => { ApiType::GenericFunction },
                    false => { ApiType::BareFunction }
                };
                if let Some(new_sequence) = 
                    self.sequence_add_fun(&api_type, api_func_index, &sequence)
                {
                    self.api_functions_visited[api_func_index] = true;
                    self.api_sequences.push(new_sequence.clone());
                    if tmp_len == max_len - 1 || (tmp_len > 0 && self.is_sequence_ended(&new_sequence)) {
                        continue;
                    }
                    if api_func_index == api_function_num - 1 {
                        take_last_one = true;
                    }
                    tmp_len += 1;
                    tmp_sequences[tmp_len - 1] = new_sequence;
                    // 此次遍历结束点
                    end_loc[tmp_len - 1] = api_func_index.try_into().unwrap();
                    break;
                }
            }
            // 找不到序列起始点，则结束
            if tmp_len == 0 && is_last_one {
                no_more = true;
            }
            // 回溯
            if tmp_len > 0 && is_last_one && !take_last_one {
                end_loc[tmp_len] = -1; 
                tmp_len -= 1;
            }
        }
    }

    // author：张彬
    pub fn dfs_with_start(&mut self, max_len: usize) {
        // 清空所有的序列
        self.api_sequences.clear();
        self.reset_visited();
        if max_len < 1 {
            return;
        }

        let api_function_num = self.api_functions.len();
        let mut start_sequences: Vec<ApiSequence> = Vec::new();
        let start_sequence = ApiSequence::new();
        for api_func_index in 0..api_function_num {
            let function = &self.api_functions[api_func_index];
            let api_type = match function._is_generic_function() {
                true => { ApiType::GenericFunction }
                false => { ApiType::BareFunction }
            };
            if let Some(new_sequence) =
                self.sequence_add_fun(&api_type, api_func_index, &start_sequence)
            {
                start_sequences.push(new_sequence.clone());
                self.api_sequences.push(new_sequence);
                self.api_functions_visited[api_func_index] = true;
            }
        }

        for sequence in &start_sequences {
            println!("start_sequence: {:?}", sequence.get_api_list());
        }

        let mut tmp_sequences: Vec<ApiSequence> = Vec::new();
        let mut tmp_sequence = ApiSequence::new();
        let mut tmp_len = 0;
        let mut no_more = false;
        let mut start_one = 0;
        let mut end_loc = -1;
        tmp_sequences.resize(max_len, tmp_sequence);
        loop {
            let mut take_last_one = false;
            let mut is_last_one = false;
            if no_more {
                break;
            }
            if tmp_len <= 0 {
                tmp_sequences[0] = start_sequences[start_one].clone();
                tmp_len += 1;
                start_one += 1;
            }
            tmp_sequence = tmp_sequences[tmp_len - 1].clone();
            println!("current_sequence: {:?}", tmp_sequence.functions);
            for api_func_index in 0..api_function_num {
                println!("tmp_len: {:?}, api_func_index: {:?}", tmp_len, api_func_index);
                if api_func_index >= api_function_num - 1 {
                    is_last_one = true;
                }
                if tmp_len < max_len - 1 && end_loc >= api_func_index.try_into().unwrap() {
                    continue;
                }
                // if self.api_functions_visited[api_func_index] {
                //     continue;
                // }
                let function = &self.api_functions[api_func_index];
                let api_type = match function._is_generic_function() {
                    true => { ApiType::GenericFunction },
                    false => { ApiType::BareFunction }
                };
                if let Some(new_sequence) = 
                    self.sequence_add_fun(&api_type, api_func_index, &tmp_sequence)
                {
                    self.api_functions_visited[api_func_index] = true;
                    self.api_sequences.push(new_sequence.clone());
                    if tmp_len == max_len - 1 || (tmp_len > 0 && self.is_sequence_ended(&new_sequence)) {
                        continue;
                    }
                    if api_func_index == api_function_num - 1 {
                        take_last_one = true;
                    }
                    tmp_len += 1;
                    tmp_sequences[tmp_len - 1] = new_sequence;
                    end_loc = api_func_index.try_into().unwrap();
                    break;
                }
            }
            // 回溯
            // 回溯后end_loc重置
            if tmp_len > 0 && is_last_one && !take_last_one {
                end_loc = end_loc - 1;
                tmp_len -= 1;
            }
            // 没有起始点，结束
            if start_one == start_sequences.len() && is_last_one {
                no_more = true;
            }
        }
    }

    //为探索比较深的路径专门进行优化
    //主要还是针对比较大的库,函数比较多的
    pub fn _try_deep_bfs(&mut self, max_sequence_number: usize) {
        //清空所有的序列
        self.api_sequences.clear();
        self.reset_visited();
        let max_len = self.api_functions.len();
        if max_len < 1 {
            return;
        }

        let api_function_num = self.api_functions.len();

        //无需加入长度为1的，从空序列开始即可，加入一个长度为0的序列作为初始
        let api_sequence = ApiSequence::new();
        self.api_sequences.push(api_sequence);

        let mut already_covered_nodes = HashSet::new();
        let mut already_covered_edges = HashSet::new();
        //接下来开始从长度1一直到max_len遍历
        for len in 0..max_len {
            let current_sequence_number = self.api_sequences.len();
            let covered_nodes = self._visited_nodes_num();
            let mut has_new_coverage_flag = false;
            if len > 2 && current_sequence_number * covered_nodes >= max_sequence_number {
                break;
            }

            let mut tmp_sequences = Vec::new();
            for sequence in &self.api_sequences {
                if self.is_sequence_ended(sequence) {
                    //如果需要引入终止函数，并且当前序列的最后一个函数是终止函数，那么就不再继续添加
                    continue;
                }
                if sequence.len() == len {
                    tmp_sequences.push(sequence.clone());
                }
            }
            for sequence in &tmp_sequences {
                //长度为len的序列，去匹配每一个函数，如果可以加入的话，就生成一个新的序列
                for api_func_index in 0..api_function_num {
                    let function = &self.api_functions[api_func_index];
                    let api_type = match function._is_generic_function() {
                        true => { ApiType::GenericFunction }
                        false => { ApiType::BareFunction }
                    };
                    if let Some(new_sequence) =
                        self.sequence_add_fun(&api_type, api_func_index, sequence)
                    {
                        let covered_nodes = new_sequence._get_contained_api_functions();
                        for covered_node in &covered_nodes {
                            if !already_covered_nodes.contains(covered_node) {
                                already_covered_nodes.insert(*covered_node);
                                has_new_coverage_flag = true;
                            }
                        }

                        let covered_edges = &new_sequence._covered_dependencies;
                        for covered_edge in covered_edges {
                            if !already_covered_edges.contains(covered_edge) {
                                already_covered_edges.insert(*covered_edge);
                                has_new_coverage_flag = true;
                            }
                        }

                        self.api_sequences.push(new_sequence);
                        self.api_functions_visited[api_func_index] = true;
                    }
                }
            }
            if !has_new_coverage_flag {
                println!("forward bfs can not find more.");
                break;
            }
        }
        println!("bfs finished, total seq: {}", self.api_sequences.len());
    }

    pub fn random_walk(&mut self, max_size: usize, stop_at_end_function: bool, max_depth: usize) {
        self.api_sequences.clear();
        self.reset_visited();

        //没有函数的话，直接return
        if self.api_functions.len() <= 0 {
            return;
        }

        //加入一个长度为0的序列
        let api_sequence = ApiSequence::new();
        self.api_sequences.push(api_sequence);

        //start random work
        let function_len = self.api_functions.len();
        let mut rng = rand::thread_rng();
        for i in 0..max_size {
            let copy_sequences = self.api_sequences.clone();
            let current_sequence_len = self.api_sequences.len();
            let chosen_sequence_index = rng.gen_range(0, current_sequence_len);
            let chosen_sequence = &copy_sequences[chosen_sequence_index];
            //如果需要在终止节点处停止
            if stop_at_end_function && self.is_sequence_ended(&chosen_sequence) {
                continue;
            }
            if max_depth > 0 && chosen_sequence.len() >= max_depth {
                continue;
            }
            let chosen_fun_index = rng.gen_range(0, function_len);
            //let chosen_fun = &self.api_functions[chosen_fun_index];
            let function = &self.api_functions[chosen_fun_index];
            let api_type = match function._is_generic_function() {
                true => { ApiType::GenericFunction }
                false => { ApiType::BareFunction }
            };
            if let Some(new_sequence) =
                self.sequence_add_fun(&api_type, chosen_fun_index, chosen_sequence)
            {
                self.api_sequences.push(new_sequence);
                self.api_functions_visited[chosen_fun_index] = true;

                //如果全都已经访问过，直接退出
                if self.check_all_visited() {
                    println!("random run {} times", i);
                    //return;
                }
            }
        }
    }

    pub fn _choose_candidate_sequence_for_merge(&self) -> Vec<usize> {
        let mut res = Vec::new();
        let all_sequence_number = self.api_sequences.len();
        for i in 0..all_sequence_number {
            let api_sequence = &self.api_sequences[i];
            let dead_code = api_sequence._dead_code(self);
            let api_sequence_len = api_sequence.len();
            if self.is_sequence_ended(api_sequence) {
                //如果当前序列已经结束
                continue;
            }
            if api_sequence_len <= 0 {
                continue;
            } else if api_sequence_len == 1 {
                res.push(i);
            } else {
                let mut dead_code_flag = true;
                for j in 0..api_sequence_len - 1 {
                    if !dead_code[j] {
                        dead_code_flag = false;
                        break;
                    }
                }
                if !dead_code_flag {
                    res.push(i);
                }
            }
        }
        res
    }

    pub fn _try_to_cover_unvisited_nodes(&mut self) {
        //println!("try to cover more nodes");
        let mut apis_covered_by_reverse_search = 0;
        let mut unvisited_nodes = HashSet::new();
        let api_fun_number = self.api_functions.len();
        for i in 0..api_fun_number {
            if !self.api_functions_visited[i] {
                unvisited_nodes.insert(i);
            }
        }
        let mut covered_node_this_iteration = HashSet::new();
        //最多循环没访问到的节点的数量
        for _ in 0..unvisited_nodes.len() {
            covered_node_this_iteration.clear();
            let candidate_sequences = self._choose_candidate_sequence_for_merge();
            //println!("sequence number, {}", self.api_sequences.len());
            //println!("candidate sequence number, {}", candidate_sequences.len());
            for unvisited_node in &unvisited_nodes {
                let unvisited_api_func = &self.api_functions[*unvisited_node];
                let inputs = &unvisited_api_func.inputs;
                let mut dependent_sequence_indexes = Vec::new();
                let mut can_be_covered_flag = true;
                let input_param_num = inputs.len();
                for i in 0..input_param_num {
                    let input_type = &inputs[i];
                    if api_util::is_fuzzable_type(input_type, &self.full_name_map) {
                        continue;
                    }
                    let mut can_find_dependency_flag = false;
                    let mut tmp_dependent_index = -1;
                    for candidate_sequence_index in &candidate_sequences {
                        let output_type = ApiType::BareFunction;
                        let input_type = ApiType::BareFunction;
                        let candidate_sequence = &self.api_sequences[*candidate_sequence_index];
                        let output_index = candidate_sequence._last_api_func_index().unwrap();
                        // 只考虑后续序列最后一个方法是否能和该方法存在依赖吗？？？
                        if let Some(_) = self.check_dependency(
                            &output_type,
                            output_index,
                            &input_type,
                            *unvisited_node,
                            i,
                        ) {
                            can_find_dependency_flag = true;
                            //dependent_sequence_indexes.push(*candidate_sequence_index);
                            tmp_dependent_index = *candidate_sequence_index as i32;

                            //prefer sequence with fuzzable inputs
                            if !candidate_sequence._has_no_fuzzables() {
                                break;
                            }
                        }
                    }
                    if !can_find_dependency_flag {
                        can_be_covered_flag = false;
                    } else {
                        dependent_sequence_indexes.push(tmp_dependent_index as usize);
                    }
                }
                if can_be_covered_flag {
                    // 该方法能够被候选序列到达
                    //println!("{:?} can be covered", unvisited_api_func.full_name);
                    let dependent_sequences: Vec<ApiSequence> = dependent_sequence_indexes
                        .into_iter()
                        .map(|index| self.api_sequences[index].clone())
                        .collect();
                    let merged_sequence = ApiSequence::_merge_sequences(&dependent_sequences);
                    let function = &self.api_functions[*unvisited_node];
                    let api_type = match function._is_generic_function() {
                        true => { ApiType::GenericFunction }
                        false => { ApiType::BareFunction }
                    };
                    if let Some(generated_sequence) =
                        self.sequence_add_fun(&api_type, *unvisited_node, &merged_sequence)
                    {
                        //println!("{}", generated_sequence._to_well_written_function(self, 0, 0));

                        self.api_sequences.push(generated_sequence);
                        self.api_functions_visited[*unvisited_node] = true;
                        covered_node_this_iteration.insert(*unvisited_node);
                        apis_covered_by_reverse_search = apis_covered_by_reverse_search + 1;
                    } else {
                        //The possible cause is there is some wrong fuzzable type
                        println!("Should not go to here. Only if algorithm error occurs");
                    }
                }
            }
            if covered_node_this_iteration.len() == 0 {
                println!("reverse search can not cover more nodes");
                break;
            } else {
                for covered_node in &covered_node_this_iteration {
                    unvisited_nodes.remove(covered_node);
                }
            }
        }

        let mut totol_sequences_number = 0;
        let mut total_length = 0;
        let mut covered_nodes = HashSet::new();
        let mut covered_edges = HashSet::new();

        for sequence in &self.api_sequences {
            if sequence._has_no_fuzzables() {
                continue;
            }
            totol_sequences_number = totol_sequences_number + 1;
            total_length = total_length + sequence.len();
            let cover_nodes = sequence._get_contained_api_functions();
            for cover_node in &cover_nodes {
                covered_nodes.insert(*cover_node);
            }

            let cover_edges = &sequence._covered_dependencies;
            for cover_edge in cover_edges {
                covered_edges.insert(*cover_edge);
            }
        }

        println!("after backward search");
        println!("targets = {}", totol_sequences_number);
        println!("total length = {}", total_length);
        let average_visit_time = (total_length as f64) / (covered_nodes.len() as f64);
        println!("average time to visit = {}", average_visit_time);
        println!("edge covered by reverse search = {}", covered_edges.len());

        //println!("There are total {} APIs covered by reverse search", apis_covered_by_reverse_search);
    }

    pub fn _naive_choose_sequence(&self, max_sequence_size: usize) -> Vec<ApiSequence> {
        println!("-----------STATISTICS-----------");
        println!("Stragety: Naive choose");
        let mut to_cover_nodes = Vec::new();
        let function_len = self.api_functions.len();
        for i in 0..function_len {
            if self.api_functions_visited[i] {
                to_cover_nodes.push(i);
            }
        }
        let to_cover_nodes_number = to_cover_nodes.len();
        println!(
            "There are total {} nodes need to be covered.",
            to_cover_nodes_number
        );

        let mut chosen_sequence_flag = Vec::new();
        let prepared_sequence_number = self.api_sequences.len();
        for _ in 0..prepared_sequence_number {
            chosen_sequence_flag.push(false);
        }

        let mut res = Vec::new();
        let mut node_candidate_sequences = HashMap::new();

        for node in &to_cover_nodes {
            node_candidate_sequences.insert(*node, Vec::new());
        }

        for i in 0..prepared_sequence_number {
            let api_sequence = &self.api_sequences[i];
            let contains_nodes = api_sequence._get_contained_api_functions();
            for node in contains_nodes {
                if let Some(v) = node_candidate_sequences.get_mut(&node) {
                    if !v.contains(&i) {
                        v.push(i);
                    }
                }
            }
        }

        let mut rng = rand::thread_rng();
        for _ in 0..max_sequence_size {
            if to_cover_nodes.len() == 0 {
                println!(
                    "all {} nodes need to be covered is covered",
                    to_cover_nodes_number
                );
                break;
            }
            //println!("need_to_cover_nodes:{:?}", to_cover_nodes);
            let next_cover_node = to_cover_nodes.first().unwrap();
            let candidate_sequences = node_candidate_sequences
                .get(next_cover_node)
                .unwrap()
                .clone();
            let unvisited_candidate_sequences = candidate_sequences
                .into_iter()
                .filter(|node| chosen_sequence_flag[*node] == false)
                .collect::<Vec<_>>();
            let candidate_number = unvisited_candidate_sequences.len();
            let random_index = rng.gen_range(0, candidate_number);
            let chosen_index = unvisited_candidate_sequences[random_index];
            //println!("randomc index{}", random_index);
            let chosen_sequence = &self.api_sequences[chosen_index];
            //println!("{:}",chosen_sequence._to_well_written_function(self, 0, 0));

            let covered_nodes = chosen_sequence._get_contained_api_functions();
            to_cover_nodes = to_cover_nodes
                .into_iter()
                .filter(|node| !covered_nodes.contains(node))
                .collect();
            chosen_sequence_flag[random_index] = true;
            res.push(chosen_sequence.clone());
        }
        println!("--------------------------------");
        res
    }

    pub fn _random_choose(&self, max_size: usize) -> Vec<ApiSequence> {
        let mut res = Vec::new();
        let mut covered_nodes = HashSet::new();
        let mut covered_edges = HashSet::new();
        let mut sequence_indexes = Vec::new();

        let total_sequence_size = self.api_sequences.len();

        for i in 0..total_sequence_size {
            sequence_indexes.push(i);
        }

        let mut rng = rand::thread_rng();
        for count in 0..max_size {
            let rest_sequences_number = sequence_indexes.len();
            if rest_sequences_number <= 0 {
                break;
            }

            let chosen_index = rng.gen_range(0, rest_sequences_number);
            let sequence_index = sequence_indexes[chosen_index];

            let sequence = &self.api_sequences[sequence_index];
            print!("No.{}: {}\n", count, sequence);
            res.push(sequence.clone());
            sequence_indexes.remove(chosen_index);

            for covered_node in sequence._get_contained_api_functions() {
                covered_nodes.insert(covered_node);
            }

            for covered_edge in &sequence._covered_dependencies {
                covered_edges.insert(covered_edge.clone());
            }
        }

        println!("-----------STATISTICS-----------");
        println!("Stragety: Random choose");
        println!("Total sequence num: {}", total_sequence_size);
        println!("Random selection selected {} targets", res.len());
        println!("Random selection covered {} nodes", covered_nodes.len());
        println!("Random selection covered {} edges", covered_edges.len());
        println!("--------------------------------");

        res
    }

    pub fn _all_choose(&self) -> Vec<ApiSequence> {
        // 随机并且全部序列都会生成（小于max_size）
        let mut res = Vec::new();
        let mut covered_nodes = HashSet::new();
        let mut covered_edges = HashSet::new();

        let total_sequence_size = self.api_sequences.len();
        let mut last_pattern_mark = "";
        for i in 0..total_sequence_size {
            let sequence = &self.api_sequences[i];
            res.push(sequence.clone());

            for covered_node in sequence._get_contained_api_functions() {
                covered_nodes.insert(covered_node);
            }

            for covered_edge in &sequence._covered_dependencies {
                covered_edges.insert(covered_edge.clone());
            }
            if let Some(pattern_mark) = &sequence._pattern_mark {
                if pattern_mark != last_pattern_mark {
                    last_pattern_mark = pattern_mark;
                    println!("Pattern: {}", last_pattern_mark);
                }
            }
            println!("No.{} {}", i, sequence);
        }

        println!("-----------STATISTICS-----------");
        println!("Stragety: All choose");
        println!("All selection selected {} targets", res.len());
        println!("All selection covered {} nodes", covered_nodes.len());
        println!("All selection covered {} edges", covered_edges.len());
        println!("--------------------------------");

        res
    }

    pub fn _unsafe_choose(&self) -> Vec<ApiSequence>{
        // 随机并且全部序列都会生成（小于max_size）
        let mut res = Vec::new();
        let mut covered_nodes = HashSet::new();
        let mut covered_edges = HashSet::new();

        let total_sequence_size = self.api_sequences.len();

        let mut unsafe_indexes = Vec::new();
        for i in 0..self.api_functions.len() {
            if self.api_functions[i].is_unsafe_function() {
                unsafe_indexes.push(i);
            }
        }
        // let api_functions = &self.api_functions;
        let api_functions = &self.api_functions;
        let mut unsafe_functions = Vec::new();
        let mut count = 0;
        let mut seq_len_vec = vec![0;20];
        let mut already_covered_edges = HashSet::new();
        for i in 0..total_sequence_size {
            let sequence = &self.api_sequences[i];
            let functions = &sequence.functions;
            let mut involved_unsafe = sequence.involved_unsafe_api();
            // println!("involved function: {:?}", involved_unsafe);
            unsafe_functions.append(&mut involved_unsafe);
            print!("No.{}: {}\n", count, sequence);
            res.push(sequence.clone());
            count += 1;

            for covered_node in sequence._get_contained_api_functions() {
                covered_nodes.insert(covered_node);
            }

            for covered_edge in &sequence._covered_dependencies {
                covered_edges.insert(covered_edge.clone());
            }

            let seq_len = sequence.functions.len();
            seq_len_vec[seq_len] += 1;

            // 计算edge覆盖情况
            let covered_edges = &sequence._covered_dependencies;
            //println!("covered_edges = {:?}", covered_edges);
            for cover_edge in covered_edges {
                already_covered_edges.insert(*cover_edge);
            }
        }
        let mut covered_unsafe_edge_num = 0;
        let mut total_unsafe_edge_num = 0;
        for edge in &already_covered_edges {
            if edge.1 {
                covered_unsafe_edge_num += 1;        
            }
        }
        for dep in &self.api_dependencies {
            if dep.1 {
                total_unsafe_edge_num += 1;
            }
        }
        let unsafe_api_function: HashSet<usize> = HashSet::from_iter(unsafe_functions);
        let total_dependencies_number = self.api_dependencies.len();

        println!("-----------STATISTICS-----------");
        println!("Unsafe choose");
        println!("Fuzz Target Number: {}", res.len());
        println!("API Number: {}", self.api_functions.len());
        println!("Involved API Number: {}", covered_nodes.len());
        println!("API Coverage: {}", covered_nodes.len() as f64 / self.api_functions.len() as f64);
        println!("Unsafe API Number: {}", unsafe_indexes.len());
        println!("Involved Unsafe API Number: {:?}", unsafe_api_function.len());
        println!("Unsafe API Coverage: {}", unsafe_api_function.len() as f64 / unsafe_indexes.len() as f64);
        println!("Edges Number: {}", total_dependencies_number);
        println!("Involved Edge Number: {}", already_covered_edges.len());
        println!("Edge Coverage: {}", already_covered_edges.len() as f64 / total_dependencies_number as f64);
        println!("Unsafe Edges Number: {}", total_unsafe_edge_num);
        println!("Involved Unsafe Edge Number: {}", covered_unsafe_edge_num);
        println!("Unsafe Edge Coverage: {}", covered_unsafe_edge_num as f64 / total_unsafe_edge_num as f64);
        // for edge in &already_covered_edges {
        //     println!("Dependency: {:?}\n{:#?}", edge, self.api_dependencies[edge.0]);
        // }
        println!("Len\tNum");
        for len in 0..seq_len_vec.len() {
            if seq_len_vec[len] > 0 {
                println!("{}\t{}", len, seq_len_vec[len]);
            }
        }
        // println!("All selection covered {} edges", covered_edges.len());
        println!("--------------------------------");

        res
    }

    pub fn _first_choose(&self, max_size: usize) -> Vec<ApiSequence> {
        let mut res = Vec::new();
        let mut covered_nodes = HashSet::new();
        let mut covered_edges = HashSet::new();

        let total_sequence_size = self.api_sequences.len();

        for index in 0..total_sequence_size {
            let sequence = &self.api_sequences[index];
            if sequence._has_no_fuzzables() {
                continue;
            }
            res.push(sequence.clone());

            for covered_node in sequence._get_contained_api_functions() {
                covered_nodes.insert(covered_node);
            }

            for covered_edge in &sequence._covered_dependencies {
                covered_edges.insert(covered_edge.clone());
            }

            if res.len() >= max_size {
                break;
            }
        }

        println!("-----------STATISTICS-----------");
        println!("Stragety: First choose");
        println!("Random walk selected {} targets", res.len());
        println!("Random walk covered {} nodes", covered_nodes.len());
        println!("Random walk covered {} edges", covered_edges.len());
        println!("--------------------------------");

        res
    }

    pub fn _heuristic_choose(
        &self,
        max_size: usize,
        stop_at_visit_all_nodes: bool,
    ) -> Vec<ApiSequence> {
        let mut res = Vec::new();
        let mut to_cover_nodes = Vec::new();

        let mut fixed_covered_nodes = HashSet::new();
        for fixed_sequence in &self.api_sequences {
            //let covered_nodes = fixed_sequence._get_contained_api_functions();
            //for covered_node in &covered_nodes {
            //    fixed_covered_nodes.insert(*covered_node);
            //}
            
            if !fixed_sequence._has_no_fuzzables()
                && !fixed_sequence._contains_dead_code_except_last_one(self)
            {
                let covered_nodes = fixed_sequence._get_contained_api_functions();
                for covered_node in &covered_nodes {
                    fixed_covered_nodes.insert(*covered_node);
                }
            }
        }

        for fixed_covered_node in fixed_covered_nodes {
            to_cover_nodes.push(fixed_covered_node);
        }

        let to_cover_nodes_number = to_cover_nodes.len();
        //println!("There are total {} nodes need to be covered.", to_cover_nodes_number);
        let to_cover_dependency_number = self.api_dependencies.len();
        //println!("There are total {} edges need to be covered.", to_cover_dependency_number);
        let total_sequence_number = self.api_sequences.len();

        //println!("There are toatl {} sequences.", total_sequence_number);
        let mut valid_fuzz_sequence_count = 0;
        for sequence in &self.api_sequences {
            if !sequence._has_no_fuzzables() && !sequence._contains_dead_code_except_last_one(self)
            {
                valid_fuzz_sequence_count = valid_fuzz_sequence_count + 1;
            }
        }
        //println!("There are toatl {} valid sequences for fuzz.", valid_fuzz_sequence_count);
        if valid_fuzz_sequence_count <= 0 {
            return res;
        }

        let mut already_covered_nodes = HashSet::new();
        let mut already_covered_edges = HashSet::new();
        let mut already_chosen_sequences = HashSet::new();
        let mut sorted_chosen_sequences = Vec::new();
        let mut dynamic_fuzzable_length_sequences_count = 0;
        let mut fixed_fuzzale_length_sequences_count = 0;

        let mut try_to_find_dynamic_length_flag = true;
        for _ in 0..max_size + 1 {
            let mut current_chosen_sequence_index = 0;
            let mut current_max_covered_nodes = 0;
            let mut current_max_covered_edges = 0;
            let mut current_chosen_sequence_len = 0;

            for j in 0..total_sequence_number {
                let api_sequence = &self.api_sequences[j];
                if already_chosen_sequences.contains(&j) {
                    // println!("Filter 1: {}", api_sequence);
                    continue;
                }

                if api_sequence._has_no_fuzzables()
                    || api_sequence._contains_dead_code_except_last_one(self)
                {
                    // println!("Filter 2: {}", api_sequence);
                    continue;
                }

                if try_to_find_dynamic_length_flag && api_sequence._is_fuzzables_fixed_length() {
                    //优先寻找fuzzable部分具有动态长度的情况
                    // println!("Filter 3: {}", api_sequence);
                    continue;
                }

                if !try_to_find_dynamic_length_flag && !api_sequence._is_fuzzables_fixed_length() {
                    //再寻找fuzzable部分具有静态长度的情况
                    // println!("Filter 4: {}", api_sequence);
                    continue;
                }

                let covered_nodes = api_sequence._get_contained_api_functions();
                let mut uncovered_nodes_by_former_sequence_count = 0;
                for covered_node in &covered_nodes {
                    if !already_covered_nodes.contains(covered_node) {
                        uncovered_nodes_by_former_sequence_count += 1;
                    }
                }

                if uncovered_nodes_by_former_sequence_count < current_max_covered_nodes {
                    // println!("Filter 5: {}", api_sequence);
                    continue;
                }
                let covered_edges = &api_sequence._covered_dependencies;
                let mut uncovered_edges_by_former_sequence_count = 0;
                for covered_edge in covered_edges {
                    if !already_covered_edges.contains(covered_edge) {
                        uncovered_edges_by_former_sequence_count =
                            uncovered_edges_by_former_sequence_count + 1;
                    }
                }
                if uncovered_nodes_by_former_sequence_count == current_max_covered_nodes
                    && uncovered_edges_by_former_sequence_count < current_max_covered_edges
                {
                    // println!("Filter 6: {}", api_sequence);
                    continue;
                }
                let sequence_len = api_sequence.len();
                if (uncovered_nodes_by_former_sequence_count > current_max_covered_nodes)
                    || (uncovered_nodes_by_former_sequence_count == current_max_covered_nodes
                        && uncovered_edges_by_former_sequence_count > current_max_covered_edges)
                    || (uncovered_nodes_by_former_sequence_count == current_max_covered_nodes
                        && uncovered_edges_by_former_sequence_count == current_max_covered_edges
                        && sequence_len < current_chosen_sequence_len)
                {
                    current_chosen_sequence_index = j;
                    current_max_covered_nodes = uncovered_nodes_by_former_sequence_count;
                    current_max_covered_edges = uncovered_edges_by_former_sequence_count;
                    current_chosen_sequence_len = sequence_len;
                }
            }
            let chosen_api_seq = &self.api_sequences[current_chosen_sequence_index];
            if try_to_find_dynamic_length_flag && current_max_covered_nodes <= 0 {
                //println!("sequences with dynamic length can not cover more nodes");
                try_to_find_dynamic_length_flag = false;
                // println!("Filter 7: {}", chosen_api_seq);
                continue;
            }

            if !try_to_find_dynamic_length_flag
                && current_max_covered_edges <= 0
                && current_max_covered_nodes <= 0
            {
                //println!("can't cover more edges or nodes");
                println!("Break 1: {}", chosen_api_seq);
                break;
            }
            println!("Chosen {}: {}", already_chosen_sequences.len(), chosen_api_seq);
            already_chosen_sequences.insert(current_chosen_sequence_index);
            sorted_chosen_sequences.push(current_chosen_sequence_index);

            if try_to_find_dynamic_length_flag {
                dynamic_fuzzable_length_sequences_count =
                    dynamic_fuzzable_length_sequences_count + 1;
            } else {
                fixed_fuzzale_length_sequences_count = fixed_fuzzale_length_sequences_count + 1;
            }

            let chosen_sequence = &self.api_sequences[current_chosen_sequence_index];

            let covered_nodes = chosen_sequence._get_contained_api_functions();
            for cover_node in covered_nodes {
                already_covered_nodes.insert(cover_node);
            }
            let covered_edges = &chosen_sequence._covered_dependencies;
            //println!("covered_edges = {:?}", covered_edges);
            for cover_edge in covered_edges {
                already_covered_edges.insert(*cover_edge);
            }

            if already_chosen_sequences.len() == valid_fuzz_sequence_count {
                //println!("all sequence visited");
                println!("Break 2: {}", chosen_api_seq);
                break;
            }
            if to_cover_dependency_number != 0
                && already_covered_edges.len() == to_cover_dependency_number
            {
                //println!("all edges visited");
                //should we stop at visit all edges?
                println!("Break 3: {}", chosen_api_seq);
                break;
            }
            if stop_at_visit_all_nodes 
                && already_covered_nodes.len() == to_cover_nodes_number {
                //println!("all nodes visited");
                println!("Break 4: {}", chosen_api_seq);
                break;
            }
            //println!("no fuzzable count = {}", no_fuzzable_count);
        }

        let mut sequnce_covered_by_reverse_search = 0;
        let mut max_length = 0;
        let mut count = 0;
        let mut unsafe_functions = Vec::new();
        let mut seq_len_vec = vec![0;20];
        for sequence_index in sorted_chosen_sequences {
            let api_sequence = self.api_sequences[sequence_index].clone();

            if api_sequence.len() > 3 {
                sequnce_covered_by_reverse_search = sequnce_covered_by_reverse_search + 1;
                if api_sequence.len() > max_length {
                    max_length = api_sequence.len();
                }
            }
            let mut involved_unsafe = api_sequence.involved_unsafe_api();
            // println!("involved function: {:?}", involved_unsafe);
            unsafe_functions.append(&mut involved_unsafe);
            let seq_len = api_sequence.functions.len();
            seq_len_vec[seq_len] += 1;
            println!("NO.{}: {}", count, api_sequence);
            res.push(api_sequence);
            count += 1;
        }

        println!("-----------STATISTICS-----------");
        println!("Stragety: Heuristic Choose");

        let mut valid_api_number = 0;
        for api_function_ in &self.api_functions {
            if !api_function_.filter_by_fuzzable_type(&self.full_name_map) {
                valid_api_number = valid_api_number + 1;
            } //else {
              //    println!("{}", api_function_._pretty_print(&self.full_name_map));
              //}
        }
        // Node Message
        let total_node_number = self.api_functions.len();
        let covered_node_num = already_covered_nodes.len();
        let node_coverage = (already_covered_nodes.len() as f64) / (valid_api_number as f64);
        println!("------------------------------------------");
        println!("{}\t|\t{}\t|\t{}\t", "TN", "IN", "CN");
        println!("{}\t|\t{}\t|\t{:.2}\t", total_node_number, covered_node_num, node_coverage);

        let mut unsafe_indexes = Vec::new();
        for i in 0..self.api_functions.len() {
            if self.api_functions[i].is_unsafe_function() {
                unsafe_indexes.push(i);
            }
        }
        let unsafe_total_node_number = unsafe_indexes.len();
        let unsafe_functions_set: HashSet<usize> = HashSet::from_iter(unsafe_functions);
        let unsafe_involved_node_number = unsafe_functions_set.len();
        let unsafe_node_coverage = unsafe_involved_node_number as f64 / unsafe_total_node_number as f64;
        println!("------------------------------------------");
        println!("{}\t|\t{}\t|\t{}\t", "UTN", "UIN", "UCN");
        println!("{}\t|\t{}\t|\t{:.2}\t", unsafe_total_node_number, unsafe_involved_node_number, unsafe_node_coverage);
        
        let total_edge_number = self.get_total_edge();
        let involved_edge_number = already_covered_edges.len();
        let coverage_edge = involved_edge_number as f64 / total_edge_number as f64;
        println!("------------------------------------------");
        println!("{}\t|\t{}\t|\t{}\t", "TE", "IE", "CE");
        println!("{}\t|\t{}\t|\t{:.2}\t", total_edge_number, involved_edge_number, coverage_edge);
        
        
        let mut total_unsafe_edge_num = 0;
        for dep in &self.api_dependencies {
            if dep.1 {
                total_unsafe_edge_num += 1;
            }
        }
        let mut covered_unsafe_edge_num = 0;
        for edge in &already_covered_edges {
            if edge.1 {
                covered_unsafe_edge_num += 1;        
            }
        }
        let unsafe_coverage_edge = covered_unsafe_edge_num as f64 / total_unsafe_edge_num as f64;
        println!("------------------------------------------");
        println!("{}\t|\t{}\t|\t{}\t", "UTE", "UIE", "UCE");
        println!("{}\t|\t{}\t|\t{:.2}\t", total_unsafe_edge_num, covered_unsafe_edge_num, unsafe_coverage_edge);
        println!("------------------------------------------");

        //println!("sequence with dynamic fuzzable length: {}", dynamic_fuzzable_length_sequences_count);
        //println!("sequence with fixed fuzzable length: {}",fixed_fuzzale_length_sequences_count);

        println!(
            "targets covered by reverse search: {}",
            sequnce_covered_by_reverse_search
        );
        println!("total targets: {}", res.len());
        println!("max length = {}", max_length);

        let mut total_length = 0;
        for selected_sequence in &res {
            total_length = total_length + selected_sequence.len();
        }

        println!("total length = {}", total_length);
        let average_time_to_fuzz_each_api =
            (total_length as f64) / (already_covered_nodes.len() as f64);
        println!(
            "average time to fuzz each api = {}",
            average_time_to_fuzz_each_api
        );

        println!("Length Contribution:");
        for len in 0..seq_len_vec.len() {
            if seq_len_vec[len] > 0{
                println!("{}\t{}", len, seq_len_vec[len]);
            }
        }
        // for edge in &already_covered_edges {
        //     println!("Dependency: {:?}\n{:#?}", edge, self.api_dependencies[edge.0]);
        // }
        println!("--------------------------------");

        res
    }

    pub fn _unsafe_heuristic_choose(
        &self,
        max_size: usize,
        stop_at_visit_all_nodes: bool,
    ) -> Vec<ApiSequence> {
        println!("Total {} Seq Before Unsafe Heuristic Choose", self.api_sequences.len());
        let mut res = Vec::new();
        let mut to_cover_nodes = Vec::new();

        let mut fixed_covered_nodes = HashSet::new();
        let mut valid_fuzz_sequence_count = 0;
        for i in 0..self.api_sequences.len() {
            let api_seq = &self.api_sequences[i];
            // println!("s{}: {}", i, self.api_sequences[i]);
            valid_fuzz_sequence_count = valid_fuzz_sequence_count + 1;
            let covered_nodes = api_seq._get_contained_api_functions();
            for covered_node in &covered_nodes {
                fixed_covered_nodes.insert(*covered_node);
            }
        }

        for fixed_covered_node in fixed_covered_nodes {
            to_cover_nodes.push(fixed_covered_node);
        }

        println!("To cover nodes: {:?}", to_cover_nodes);

        let to_cover_nodes_number = to_cover_nodes.len();
        //println!("There are total {} nodes need to be covered.", to_cover_nodes_number);
        let to_cover_dependency_number = self.api_dependencies.len();
        //println!("There are total {} edges need to be covered.", to_cover_dependency_number);
        let total_sequence_number = self.api_sequences.len();

        //println!("There are toatl {} sequences.", total_sequence_number);
        //println!("There are toatl {} valid sequences for fuzz.", valid_fuzz_sequence_count);
        // if valid_fuzz_sequence_count <= 0 {
        //     println!("valid fuzz sequence <= 0");
        //     return res;
        // }

        let mut already_covered_nodes = HashSet::new();
        let mut already_covered_edges = HashSet::new();
        let mut already_chosen_sequences = HashSet::new();
        let mut sorted_chosen_sequences = Vec::new();

        // let mut try_to_find_dynamic_length_flag = true;
        for _ in 0..max_size + 1 {
            let mut current_chosen_sequence_index = 9999999;
            let mut current_max_covered_nodes = 0;
            let mut unsafe_current_max_covered_nodes = 0;
            let mut current_max_covered_edges = 0;
            let mut unsafe_current_max_covered_edges = 0;
            let mut current_chosen_sequence_len = 0;
            // let mut last_unsafe_nodes = 0;

            for j in 0..total_sequence_number {
                let api_sequence = &self.api_sequences[j];
                let mut unsafe_nodes = HashSet::new();
                if already_chosen_sequences.contains(&j) {
                    // println!("Filter 1: {}", api_sequence);
                    continue;
                }
                // if !self.check_syntax(&api_sequence) {
                //     println!("Filter 2: {}", api_sequence);
                //     continue;
                // }

                // if api_sequence._has_no_fuzzables()
                //     || api_sequence._contains_dead_code_except_last_one(self)
                // {
                //     // println!("Filter 2: {}", api_sequence);
                //     continue;
                // }

                let covered_nodes = api_sequence._get_contained_api_functions();
                let unsafe_covered_nodes = api_sequence._get_contained_unsafe_api_functions(self);
                let mut uncovered_nodes_by_former_sequence_count = 0;
                let mut unsafe_uncovered_nodes_by_former_sequence_count = 0;
                for covered_node in &covered_nodes {
                    if !already_covered_nodes.contains(covered_node) {
                        uncovered_nodes_by_former_sequence_count += 1;
                    }
                }
                for unsafe_covered_node in &unsafe_covered_nodes {
                    if !already_covered_nodes.contains(unsafe_covered_node) {
                        unsafe_uncovered_nodes_by_former_sequence_count += 1;
                    }
                    if !unsafe_nodes.contains(unsafe_covered_node) {
                        unsafe_nodes.insert(unsafe_covered_node);
                    }
                }

                let covered_edges = &api_sequence._covered_dependencies;
                let mut uncovered_edges_by_former_sequence_count = 0;
                let mut unsafe_uncovered_edges_by_former_sequence_count = 0;
                for covered_edge in covered_edges {
                    if !already_covered_edges.contains(covered_edge) {
                        uncovered_edges_by_former_sequence_count += 1;
                        if covered_edge.1 {
                            unsafe_uncovered_edges_by_former_sequence_count += 1
                        }
                    }
                }
                if uncovered_nodes_by_former_sequence_count < current_max_covered_nodes
                    && unsafe_uncovered_nodes_by_former_sequence_count < unsafe_current_max_covered_nodes 
                    && uncovered_edges_by_former_sequence_count < current_max_covered_edges
                    && unsafe_uncovered_edges_by_former_sequence_count < unsafe_current_max_covered_edges
                {
                    // println!("Filter 5: {}", api_sequence);
                    continue;
                }
                // if uncovered_nodes_by_former_sequence_count == current_max_covered_nodes
                //     && uncovered_edges_by_former_sequence_count < current_max_covered_edges
                //     && unsafe_uncovered_nodes_by_former_sequence_count < unsafe_current_max_covered_nodes
                //     && unsafe_uncovered_edges_by_former_sequence_count < unsafe_current_max_covered_edges
                // {
                //     println!("Filter 6: {}", api_sequence);
                //     continue;
                // }
                // if unsafe_uncovered_nodes_by_former_sequence_count == unsafe_current_max_covered_nodes
                //     && uncovered_edges_by_former_sequence_count < current_max_covered_edges
                //     && uncovered_nodes_by_former_sequence_count < current_max_covered_nodes
                //     && unsafe_uncovered_edges_by_former_sequence_count < unsafe_current_max_covered_edges
                // {
                //     println!("Filter 7: {}", api_sequence);
                //     continue;
                // }
                let sequence_len = api_sequence.len();
                // println!("{}, {}, {}, {}", uncovered_nodes_by_former_sequence_count, unsafe_uncovered_nodes_by_former_sequence_count, uncovered_edges_by_former_sequence_count, unsafe_uncovered_edges_by_former_sequence_count);
                if (uncovered_nodes_by_former_sequence_count > current_max_covered_nodes)
                    || (uncovered_nodes_by_former_sequence_count == current_max_covered_nodes
                        && unsafe_uncovered_nodes_by_former_sequence_count > unsafe_current_max_covered_nodes)
                    || (uncovered_nodes_by_former_sequence_count == current_max_covered_nodes
                        && unsafe_uncovered_nodes_by_former_sequence_count == unsafe_current_max_covered_nodes
                        && uncovered_edges_by_former_sequence_count > current_max_covered_edges)
                    || (uncovered_nodes_by_former_sequence_count == current_max_covered_nodes
                        && unsafe_uncovered_nodes_by_former_sequence_count == unsafe_current_max_covered_nodes
                        && uncovered_edges_by_former_sequence_count == current_max_covered_edges
                        && unsafe_uncovered_edges_by_former_sequence_count > unsafe_current_max_covered_edges)
                    || (uncovered_nodes_by_former_sequence_count == current_max_covered_nodes
                        && unsafe_uncovered_nodes_by_former_sequence_count == unsafe_current_max_covered_nodes
                        && uncovered_edges_by_former_sequence_count == current_max_covered_edges
                        && unsafe_uncovered_edges_by_former_sequence_count == unsafe_current_max_covered_edges
                        && sequence_len < current_chosen_sequence_len)
                    // || (uncovered_nodes_by_former_sequence_count == current_max_covered_nodes
                    //     && unsafe_uncovered_nodes_by_former_sequence_count == unsafe_current_max_covered_nodes
                    //     && uncovered_edges_by_former_sequence_count == current_max_covered_edges
                    //     && unsafe_uncovered_edges_by_former_sequence_count == unsafe_current_max_covered_edges
                    //     && unsafe_nodes.len() == last_unsafe_nodes
                    //     && sequence_len < current_chosen_sequence_len)
                {
                    current_chosen_sequence_index = j;
                    current_max_covered_nodes = uncovered_nodes_by_former_sequence_count;
                    current_max_covered_edges = uncovered_edges_by_former_sequence_count;
                    unsafe_current_max_covered_edges = unsafe_uncovered_edges_by_former_sequence_count;
                    unsafe_current_max_covered_nodes = unsafe_uncovered_nodes_by_former_sequence_count;
                    current_chosen_sequence_len = sequence_len;
                    // last_unsafe_nodes = unsafe_nodes.len();
                }
            }
            if current_chosen_sequence_index == 9999999 {
                // 没选到
                break;
            }
            let chosen_api_seq = &self.api_sequences[current_chosen_sequence_index];

            if current_max_covered_edges + unsafe_current_max_covered_edges <= 0
                && current_max_covered_nodes + unsafe_current_max_covered_nodes <= 0
            {
                //println!("can't cover more edges or nodes");
                println!("Break 1: {}", chosen_api_seq);
                break;
            }
            println!("Chosen {}: {}", already_chosen_sequences.len(), chosen_api_seq);
            already_chosen_sequences.insert(current_chosen_sequence_index);
            sorted_chosen_sequences.push(current_chosen_sequence_index);

            let chosen_sequence = &self.api_sequences[current_chosen_sequence_index];

            let covered_nodes = chosen_sequence._get_contained_api_functions();
            for cover_node in covered_nodes {
                already_covered_nodes.insert(cover_node);
            }
            let covered_edges = &chosen_sequence._covered_dependencies;
            //println!("covered_edges = {:?}", covered_edges);
            for cover_edge in covered_edges {
                already_covered_edges.insert(*cover_edge);
            }

            if already_chosen_sequences.len() == valid_fuzz_sequence_count {
                //println!("all sequence visited");
                println!("Break 2: {}", chosen_api_seq);
                break;
            }
            if to_cover_dependency_number != 0
                && already_covered_edges.len() == to_cover_dependency_number
            {
                //println!("all edges visited");
                //should we stop at visit all edges?
                println!("Break 3: {}", chosen_api_seq);
                break;
            }
            if stop_at_visit_all_nodes 
                && already_covered_nodes.len() == to_cover_nodes_number
                && already_covered_edges.len() == self.api_dependencies.len() {
                //println!("all nodes visited");
                println!("Break 4: {}", chosen_api_seq);
                break;
            }
            //println!("no fuzzable count = {}", no_fuzzable_count);
        }

        let mut sequnce_covered_by_reverse_search = 0;
        let mut max_length = 0;
        let mut count = 0;
        let mut unsafe_functions = Vec::new();
        let mut seq_len_vec = vec![0;20];
        for sequence_index in sorted_chosen_sequences {
            let api_sequence = self.api_sequences[sequence_index].clone();

            if api_sequence.len() > 3 {
                sequnce_covered_by_reverse_search = sequnce_covered_by_reverse_search + 1;
                if api_sequence.len() > max_length {
                    max_length = api_sequence.len();
                }
            }
            let mut involved_unsafe = api_sequence.involved_unsafe_api();
            // println!("involved function: {:?}", involved_unsafe);
            unsafe_functions.append(&mut involved_unsafe);
            let seq_len = api_sequence.functions.len();
            seq_len_vec[seq_len] += 1;
            println!("NO.{}: {}", count, api_sequence.get_sequence_string());
            // println!("{:#?}", api_sequence);
            // println!("Output Types: {}", api_sequence.calc_output_type());
            // for api_call in &api_sequence.functions {
            //     println!("output: {:#?}", api_call.output_type);
            // }
            res.push(api_sequence);
            count += 1;
        }

        println!("-----------------STATISTICS-----------------");
        println!("Stragety: Unsafe Heuristic Choose");

        let mut valid_api_number = 0;
        for api_function_ in &self.api_functions {
            if !api_function_.filter_by_fuzzable_type(&self.full_name_map) {
                valid_api_number = valid_api_number + 1;
            }
        }
        // Node Message
        // let total_node_number = self.api_functions.len();
        // let covered_node_num = already_covered_nodes.len();
        // let node_coverage = (already_covered_nodes.len() as f64) / (valid_api_number as f64);
        // println!("------------------------------------------------");
        // println!("|\t{}\t|\t{}\t|\t{}\t|", "TN", "IN", "CN");
        // println!("|\t{}\t|\t{}\t|\t{:.2}\t|", total_node_number, covered_node_num, node_coverage);

        // let mut unsafe_indexes = Vec::new();
        // for i in 0..self.api_functions.len() {
        //     if self.api_functions[i].is_unsafe_function() {
        //         unsafe_indexes.push(i);
        //     }
        // }
        // let unsafe_total_node_number = unsafe_indexes.len();
        // let unsafe_functions_set: HashSet<usize> = HashSet::from_iter(unsafe_functions);
        // let unsafe_involved_node_number = unsafe_functions_set.len();
        // let unsafe_node_coverage = unsafe_involved_node_number as f64 / unsafe_total_node_number as f64;
        // println!("------------------------------------------------");
        // println!("|\t{}\t|\t{}\t|\t{}\t|", "UTN", "UIN", "UCN");
        // println!("|\t{}\t|\t{}\t|\t{:.2}\t|", unsafe_total_node_number, unsafe_involved_node_number, unsafe_node_coverage);
        
        // let total_edge_number = self.get_total_edge();
        // let involved_edge_number = already_covered_edges.len();
        // let coverage_edge = involved_edge_number as f64 / total_edge_number as f64;
        // println!("------------------------------------------------");
        // println!("|\t{}\t|\t{}\t|\t{}\t|", "TE", "IE", "CE");
        // println!("|\t{}\t|\t{}\t|\t{:.2}\t|", total_edge_number, involved_edge_number, coverage_edge);
        
        
        // let total_unsafe_edge_num = self.get_total_unsafe_edge();
        // let mut covered_unsafe_edge_num = 0;
        // for edge in &already_covered_edges {
        //     if edge.1 {
        //         covered_unsafe_edge_num += 1;        
        //     }
        // }
        // let unsafe_coverage_edge = covered_unsafe_edge_num as f64 / total_unsafe_edge_num as f64;
        // println!("------------------------------------------------");
        // println!("|\t{}\t|\t{}\t|\t{}\t|", "UTE", "UIE", "UCE");
        // println!("|\t{}\t|\t{}\t|\t{:.2}\t|", total_unsafe_edge_num, covered_unsafe_edge_num, unsafe_coverage_edge);
        // println!("------------------------------------------------");

        //println!("sequence with dynamic fuzzable length: {}", dynamic_fuzzable_length_sequences_count);
        //println!("sequence with fixed fuzzable length: {}",fixed_fuzzale_length_sequences_count);

        println!(
            "targets covered by reverse search: {}",
            sequnce_covered_by_reverse_search
        );
        println!("total targets: {}", res.len());
        println!("max length = {}", max_length);

        let mut total_length = 0;
        for selected_sequence in &res {
            total_length = total_length + selected_sequence.len();
        }

        println!("total length = {}", total_length);
        let average_time_to_fuzz_each_api =
            (total_length as f64) / (already_covered_nodes.len() as f64);
        println!(
            "average time to fuzz each api = {}",
            average_time_to_fuzz_each_api
        );

        println!("Length Contribution:");
        for len in 0..seq_len_vec.len() {
            if seq_len_vec[len] > 0{
                println!("{}\t{}", len, seq_len_vec[len]);
            }
        }
        // for edge in &already_covered_edges {
        //     println!("Dependency: {:?}\n{:#?}", edge, self.api_dependencies[edge.0]);
        // }
        println!("--------------------------------------");
        
        println!("COVERED NODES: ");
        let mut covered_nodes = already_covered_nodes.into_iter().collect::<Vec<_>>();
        covered_nodes.sort();
        println!("{:?}", covered_nodes);

        println!("COVERED EDGES: ");
        let mut covered_edges = already_covered_edges.into_iter().collect::<Vec<_>>();
        covered_edges.sort();
        for edge in &covered_edges {
            let output_func = self.api_dependencies[edge.0].0.output_fun.1;
            let input_func = self.api_dependencies[edge.0].0.input_fun.1;
            let param_index = self.api_dependencies[edge.0].0.input_param_index;
            print!("{}->{}({}), ", output_func, input_func, param_index);
        }
        println!("");

        self.create_graph_dot(&covered_nodes, &covered_edges);

        res
    }

    pub fn _pattern_choose(&self, max_size: usize,) -> Vec<ApiSequence> {
        // 随机并且全部序列都会生成（小于max_size）
        let mut res = Vec::new();
        let mut covered_nodes = HashSet::new();
        let mut covered_edges = HashSet::new();

        let total_sequence_size = self.api_sequences.len();
        let mut last_pattern_mark = "";
        for i in 0..total_sequence_size {
            let sequence = &self.api_sequences[i];
            res.push(sequence.clone());

            for covered_node in sequence._get_contained_api_functions() {
                covered_nodes.insert(covered_node);
            }

            for covered_edge in &sequence._covered_dependencies {
                covered_edges.insert(covered_edge.clone());
            }
            if let Some(pattern_mark) = &sequence._pattern_mark {
                if pattern_mark != last_pattern_mark {
                    last_pattern_mark = pattern_mark;
                    println!("Pattern: {}", last_pattern_mark);
                }
            }
            println!("No.{} {}", i, sequence);
        }

        println!("-----------STATISTICS-----------");
        println!("Stragety: All choose");
        println!("All selection selected {} targets", res.len());
        println!("All selection covered {} nodes", covered_nodes.len());
        println!("All selection covered {} edges", covered_edges.len());
        println!("--------------------------------");

        res
    }

    pub fn _new_heuristic_choose(
        &self,
        max_size: usize,
        stop_at_visit_all_nodes: bool,
    ) -> Vec<ApiSequence> {
        Vec::new()
    }

    // 分析生成的sequence序列
    // 包括以下信息：
    // (UNSAFE)API覆盖情况
    // (UNSAFE)EDGE覆盖情况
    // 序列数量
    pub fn evaluate_sequences(&self) {
        let mut involved_apis = Vec::new();
        let mut involved_edges = Vec::new();
        for seq in &self.api_sequences {
            // 更新api覆盖情况
            let covered_apis = seq._get_contained_api_functions();
            for covered_api in &covered_apis{
                if !involved_apis.contains(covered_api) {
                    involved_apis.push(*covered_api);
                }
            }

            // 更新edge覆盖情况
            let covered_edges = &seq._covered_dependencies;
            //println!("covered_edges = {:?}", covered_edges);
            for covered_edge in covered_edges {
                if !involved_edges.contains(covered_edge) {
                    involved_edges.push(*covered_edge);
                }
            }
        }
        println!("-----------------STATISTICS-----------------");
        let total_api_num = self.api_functions.len();
        let involved_api_num = involved_apis.len();
        let api_coverage = involved_api_num as f64 / total_api_num as f64;
        println!("------------------------------------------------");
        println!("|\t{}\t|\t{}\t|\t{}\t|", "TN", "IN", "CN");
        println!("|\t{}\t|\t{}\t|\t{:.2}\t|", total_api_num, involved_api_num, api_coverage);

        let unsafe_total_api_num = &self.api_functions
            .clone()
            .into_iter()
            .filter(|func| func.is_unsafe_function())
            .collect::<Vec<_>>()
            .len();
        let unsafe_involved_api_num = involved_apis
            .into_iter()
            .filter(|i| self.api_functions[*i].is_unsafe_function())
            .collect::<Vec<_>>()
            .len();
        let unsafe_api_coverage = unsafe_involved_api_num as f64 / *unsafe_total_api_num as f64;
        println!("------------------------------------------------");
        println!("|\t{}\t|\t{}\t|\t{}\t|", "UTN", "UIN", "UCN");
        println!("|\t{}\t|\t{}\t|\t{:.2}\t|", unsafe_total_api_num, unsafe_involved_api_num, unsafe_api_coverage);

        let total_edge_num = self.get_total_edge();
        let involved_edge_num = involved_edges.len();
        let coverage_edge = involved_edge_num as f64 / total_edge_num as f64;
        println!("------------------------------------------------");
        println!("|\t{}\t|\t{}\t|\t{}\t|", "TE", "IE", "CE");
        println!("|\t{}\t|\t{}\t|\t{:.2}\t|", total_edge_num, involved_edge_num, coverage_edge);

        let unsafe_total_edge_num = self.get_total_unsafe_edge();
        let unsafe_involved_edge_num = involved_edges
            .into_iter()
            .filter(|dep| dep.1)
            .collect::<Vec<_>>()
            .len();
        let unsafe_coverage_edge = unsafe_involved_edge_num as f64 / unsafe_total_edge_num as f64;
        println!("------------------------------------------------");
        println!("|\t{}\t|\t{}\t|\t{}\t|", "UTE", "UIE", "UCE");
        println!("|\t{}\t|\t{}\t|\t{:.2}\t|", unsafe_total_edge_num, unsafe_involved_edge_num, unsafe_coverage_edge);
        println!("------------------------------------------------");
    }

    // 判断一个函数能否加入给定的序列中,如果可以加入，返回Some(new_sequence),new_sequence是将新的调用加进去之后的情况，否则返回None
    pub fn sequence_add_fun(
        &mut self,
        api_type: &ApiType,
        input_fun_index: usize,
        sequence: &ApiSequence,
    ) -> Option<ApiSequence> {
        // 因为T->T的存在，因此返回的可能是一个序列的数组
        // 或者找一个T
        // let mut res_sequences = Vec::new();
        //判断一个给定的函数能否加入到一个sequence中去
        println!("\n>>>>> SEQUENCE CALL FUNCTION");
        match api_type {
            ApiType::BareFunction | ApiType::GenericFunction => {
                // println!("original seq: {}", sequence);
                let mut new_sequence = sequence.clone();
                let generic_map = sequence.generic_map.clone();
                let input_function = &self.api_functions[input_fun_index];
                // println!("Target function: {}", input_function.full_name);
                let mut api_call = ApiCall::new(input_fun_index, input_function.output.clone(), input_function._is_generic_function());
                match &input_function.output {
                    Some(ty) => {
                        if !api_util::_is_end_type(&ty, &self.full_name_map) {
                            api_call.set_output_type(Some(ty.clone()));
                        }
                    },
                    None => {}
                }
                let mut _moved_indexes = HashSet::new(); //用来保存发生move的那些语句的index
                                                         //用来保存会被多次可变引用的情况
                let mut _multi_mut = HashSet::new();
                let mut _immutable_borrow = HashSet::new();

                let is_unsafe_input_function = input_function.is_unsafe_function();
                // 如果函数含有unsafe, getRawPtr, drop 操作，则标记api_call
                if !input_function.unsafe_info.is_empty(){
                    api_call.is_unsafe = true;
                }
                if !input_function.rawptr_info.is_empty(){
                    api_call.is_get_raw_ptr = true;
                }
                if !input_function.drop_info.is_empty(){
                    api_call.is_drop = true;
                }
                if !input_function.mutate_info.is_empty(){
                    api_call.is_mutate = true;
                }
                // 如果是个unsafe函数，给sequence添加unsafe标记
                if input_function._unsafe_tag._is_unsafe() {
                    new_sequence.set_unsafe();
                }
                if input_function._trait_full_path.is_some() {
                    let trait_full_path = input_function._trait_full_path.as_ref().unwrap();
                    new_sequence.add_trait(trait_full_path);
                }
                let input_params = &input_function.inputs;
                let input_params_num = input_params.len();
                if input_params_num == 0 {
                    //无需输入参数，直接是可满足的
                    new_sequence._add_fn(api_call);
                    return Some(new_sequence);
                }
                for i in 0..input_params_num {
                    println!(">>> {}-{} <<<", input_fun_index, i);
                    // current_ty 只是用于查看是否可以fuzzable
                    // 
                    // let current_ty = self.get_real_type(input_function, &input_params[i]);
                    // 先不考虑自己生成preluded_type->T
                    let current_ty = &input_params[i];
                    // println!("{}-{} current type: {:#?}", input_function.full_name, i, current_ty);
                    println!("i). CALL WITH FUZZABLE PARAMETER");
                    if api_util::is_fuzzable_type(&current_ty, &self.full_name_map) {
                        //如果当前参数是fuzzable的
                        let current_fuzzable_index = new_sequence.fuzzable_params.len();
                        let fuzzable_call_type =
                            fuzzable_type::fuzzable_call_type_by_clean_type(&current_ty, &self.full_name_map);
                        let (fuzzable_type, call_type) =
                            fuzzable_call_type.generate_fuzzable_type_and_call_type();

                        //如果出现了下面这段话，说明出现了Fuzzable参数但不知道如何参数化的
                        //典型例子是tuple里面出现了引用（&usize），这种情况不再去寻找dependency，直接返回无法添加即可
                        match &fuzzable_type {
                            FuzzableType::NoFuzzable => {
                                return None;
                            }
                            _ => { }
                        }

                        //判断要不要加mut tag
                        if api_util::_need_mut_tag(&call_type) {
                            new_sequence._insert_fuzzable_mut_tag(current_fuzzable_index);
                        }

                        //添加到sequence中去
                        new_sequence.fuzzable_params.push(fuzzable_type);
                        api_call._add_param(
                            ParamType::_FuzzableType,
                            current_fuzzable_index,
                            call_type,
                        );
                        println!("PARAMETER FUZZABLE");
                        continue;
                    }
                    //如果当前参数不是fuzzable的，那么就去api sequence寻找是否有这个依赖
                    //TODO:处理move的情况
                    println!("ii). CALL WITH ORIGIN SEQUENCE");
                    let input_type = &input_function.inputs[i];
                    let functions_in_sequence_len = sequence.functions.len();
                    let mut dependency_flag = false;
                    for function_index in 0..functions_in_sequence_len {
                        //如果这个sequence里面的该函数返回值已经被move掉了，那么就跳过，不再能被使用了
                        if new_sequence._is_moved(function_index)
                            || _moved_indexes.contains(&function_index)
                        {
                            continue;
                        }
                        let found_function = &new_sequence.functions[function_index];
                        let (call_api_type, index) = &found_function.func;
                        let output_function = &self.api_functions[*index];
                        let is_unsafe_found_function = output_function.is_unsafe_function();
                        if let Some(dependency_index) =
                            self.check_dependency(call_api_type, *index, api_type, input_fun_index, i)
                        {
                            let output_type = match &output_function.output {
                                Some(type_) => type_,
                                None => {
                                    println!("function: {}", output_function.full_name);
                                    panic!("can not get output after check dependency");
                                }
                            };
                            let dependency_ = self.api_dependencies[dependency_index].0.clone();
                            // println!("check dependency pass!: {:#?}", dependency_);
                            let interest_parameters = &dependency_.parameters;
                            let (is_call_parameter_generic, is_called_paramter_generic) = (output_function._is_generic_output(), input_function._is_generic_input(i));
                            // GenericCall
                            match (is_call_parameter_generic, is_called_paramter_generic) {
                                (false, false) => {
                                    // Do Nothing
                                },
                                (true, true) => {
                                    let mut output_generic_path_name = String::new();
                                    let mut input_generic_path_name = String::new();
                                    let output_generic_parmeter = match api_parameter::get_generic_name(output_type) {
                                        Some(symbol) => {
                                            output_generic_path_name = output_function.get_generic_path_by_symbol(&symbol).unwrap();
                                            let parameter = new_sequence.get_generic_parameter(&output_generic_path_name);
                                            parameter
                                        }
                                        None => None
                                    };
                                    let input_generic_parmeter = match api_parameter::get_generic_name(input_type) {
                                        Some(symbol) => {
                                            input_generic_path_name = input_function.get_generic_path_by_symbol(&symbol).unwrap();
                                            let parameter = new_sequence.get_generic_parameter(&input_generic_path_name);
                                            parameter
                                        }
                                        None => None
                                    };

                                    match (output_generic_parmeter, input_generic_parmeter) {
                                        // ouput(S) -> input(S)
                                        (Some(output_parameter), Some(input_parameter)) => {
                                            if output_parameter.as_string() != input_parameter.as_string(){
                                                continue;
                                            }
                                        },
                                        // ouput(S) -> input(T)
                                        (Some(output_parameter), None) => {
                                            let mut dependency_flag = false;
                                            for param in interest_parameters {
                                                if param.as_string() == output_parameter.as_string() {
                                                    dependency_flag = true;
                                                    new_sequence.add_generic_info(&input_generic_path_name, param.clone());
                                                    break;
                                                }
                                            }
                                            if !dependency_flag { continue; }
                                        },
                                        // ouput(T) -> input(S)
                                        (None, Some(input_parameter)) => {
                                            let mut dependency_flag = false;
                                            for param in interest_parameters {
                                                if param.as_string() == input_parameter.as_string() {
                                                    dependency_flag = true;
                                                    new_sequence.add_generic_info(&output_generic_path_name, param.clone());
                                                    break;
                                                }
                                            }
                                            if !dependency_flag { continue; }
                                        },
                                        // ouput(T) -> input(T)
                                        (None, None) => {
                                            // 在interestParameter选一个Parameter
                                            // 暂时采用随机选取
                                            // TODO: Parameter评分，基于sequence还是function
                                            let mut rng = rand::thread_rng();
                                            let random_num = rng.gen_range(0, interest_parameters.len());
                                            let param = &interest_parameters[random_num];
                                            new_sequence.add_generic_info(&output_generic_path_name, param.clone());
                                            new_sequence.add_generic_info(&input_generic_path_name, param.clone());
                                        },
                                    }
                                },
                                (false, true) => {
                                    // SpecialType -> GenericType
                                    if let Some(symbol) = api_parameter::get_generic_name(input_type) {
                                        // println!("dependency: {:#?}", dependency_);
                                        assert_eq!(interest_parameters.len(), 1);
                                        let generic_path_name = input_function.get_generic_path_by_symbol(&symbol).unwrap();
                                        if let Some(generic_parameter) = new_sequence.get_generic_parameter(&generic_path_name) {
                                            // seq里有泛型的具体类型
                                            // 则只有一个选择，看能不能满足input
                                            if interest_parameters[0].as_string() != generic_parameter.as_string() {
                                                continue;
                                            }
                                        } else {
                                            // 没有则添加具体类型
                                            new_sequence.add_generic_info(&generic_path_name, interest_parameters[0].clone());
                                        }
                                    } else { }
                                },
                                (true, false) => {
                                    // GenericType -> SpecialType
                                    // 确保GenericType和SpecialType的关系存在
                                    if let Some(symbol) = api_parameter::get_generic_name(output_type) {
                                        assert_eq!(interest_parameters.len(), 1);
                                        let generic_path_name = output_function.get_generic_path_by_symbol(&symbol).unwrap();
                                        if let Some(generic_parameter) = new_sequence.get_generic_parameter(&generic_path_name) {
                                            // seq里有泛型的具体类型
                                            // 则只有一个选择，看能不能满足input
                                            if interest_parameters[0].as_string() != generic_parameter.as_string() {
                                                continue;
                                            }
                                        } else {
                                            // 没有则添加具体类型
                                            new_sequence.add_generic_info(&generic_path_name, interest_parameters[0].clone());
                                        }
                                    } else { }
                                    
                                },
                            }
                            
                            // for parameter in interest_parameters {
                            //     let parameter_name = parameter.as_string();
                            //     if let Some(symbol) = api_structure::get_generic_name(&output_type) {
                            //         let generic_path_name = found_function.get_generic_path_by_symbol(symbol);
                            //         new_sequence.add_generic_info(generic_path_name, parameter);
                            //     }
                            // }
                            //将覆盖到的边加入到新的sequence中去
                            new_sequence._add_dependency((dependency_index, is_unsafe_found_function || is_unsafe_input_function));
                            //找到了依赖，当前参数是可以被满足的，设置flag并退出循环
                            dependency_flag = true;
                            //如果满足move发生的条件，那么
                            if api_util::_move_condition(&current_ty, &dependency_.call_type) {
                                if _multi_mut.contains(&function_index)
                                    || _immutable_borrow.contains(&function_index)
                                {
                                    dependency_flag = false;
                                    continue;
                                } else {
                                    _moved_indexes.insert(function_index);
                                }
                            }
                            //如果当前调用是可变借用
                            if api_util::_is_mutable_borrow_occurs(
                                &current_ty,
                                &dependency_.call_type,
                            ) {
                                //如果之前已经被借用过了
                                if _multi_mut.contains(&function_index)
                                    || _immutable_borrow.contains(&function_index)
                                {
                                    dependency_flag = false;
                                    continue;
                                } else {
                                    _multi_mut.insert(function_index);
                                }
                            }
                            //如果当前调用是引用，且之前已经被可变引用过，那么这个引用是非法的
                            if api_util::_is_immutable_borrow_occurs(
                                &current_ty,
                                &dependency_.call_type,
                            ) {
                                if _multi_mut.contains(&function_index) {
                                    dependency_flag = false;
                                    continue;
                                } else {
                                    _immutable_borrow.insert(function_index);
                                }
                            }
                            //参数需要加mut 标记的话
                            if api_util::_need_mut_tag(&dependency_.call_type) {
                                new_sequence._insert_function_mut_tag(function_index);
                            }
                            api_call._add_param(
                                ParamType::_FunctionReturn,
                                function_index,
                                dependency_.call_type,
                            );
                            // syntax检查
                            println!("{} -> {}-{}", function_index, input_fun_index, i);
                            let mut test_sequence = new_sequence.clone();
                            test_sequence._add_fn(api_call.clone());
                            match self.sequence_syntax_analyse(&test_sequence) {
                                true => {
                                    println!("SEQUENCE CALL FUNCTION SUCCESS <<<<<\n");
                                    break;
                                }
                                false => {
                                    dependency_flag = false;
                                    println!("SEQUENCE CALL FUNCTION FAILED (SYNTAX) <<<<<\n");
                                    continue;
                                }
                            }
                            // println!("PARAMETER SUPPORTED");
                        } else {
                            println!("PARAMETER CAN'T SUPPORTED");
                            // 没有可满足的dependency
                            // 更新refuse_fun_indexes
                            // println!("cann't find dependency");
                            if let Some(refuse_fun_indexes) = self.refuse_api_map.get_mut(&input_fun_index){
                                refuse_fun_indexes[i].insert(*index);
                            }
                        }
                    }
                    if !dependency_flag {
                        //如果这个参数没有寻找到依赖，则这个函数不可以被加入到序列中
                        println!("SEQUENCE CALL FUNCTION FAILED <<<<<\n");
                        return None;
                    }
                }
                //所有参数都可以找到依赖，那么这个函数就可以加入序列
                new_sequence._add_fn(api_call);
                if new_sequence._contains_multi_dynamic_length_fuzzable() {
                    //如果新生成的序列包含多维可变的参数，就不把这个序列加进去
                    println!("SEQUENCE CALL FUNCTION FAILED (DYNAMIC LENGTH FUZZABLE) <<<<<\n");
                    return None;
                }
                return Some(new_sequence);
            }
            ApiType::ControlStart | ApiType::ScopeEnd | ApiType::UseParam => {
                println!("SEQUENCE CALL FUNCTION FAILED (APITYPE) <<<<<\n");
                return None;
            }
            
        }
    }

    //判断一个依赖是否存在,存在的话返回Some(ApiDependency),否则返回None
    pub fn check_dependency(
        &self,
        output_type: &ApiType,
        output_index: usize,
        input_type: &ApiType,
        input_index: usize,
        input_param_index_: usize,
    ) -> Option<usize> {
        println!("\n>>> CHECK DEPENDENCY\n{}->{}-{}", 
            self.api_functions[output_index].full_name, 
            self.api_functions[input_index].full_name,
            input_param_index_
        );
        let dependency_num = self.api_dependencies.len();
        let mut tmp_dependency = ApiDependency {
            output_fun: (*output_type, output_index),
            input_fun: (*input_type, input_index),
            input_param_index: input_param_index_,
            call_type: CallType::_NotCompatible,
            parameters: Vec::new(),
        };
        // println!("tmp: {:#?}", tmp_dependency);
        for index in 0..dependency_num {
            let dependency = &self.api_dependencies[index].0;
            tmp_dependency.call_type = dependency.call_type.clone();
            //TODO:直接比较每一项内容是否可以节省点时间？
            if tmp_dependency.is_same_dependency(dependency) {
                //存在依赖
                println!("SUCCESS <<<\n");
                return Some(index);
            }
        }
        //没找到依赖
        println!("FAILED<<< \n");
        return None;
    }

    //判断一个调用序列是否已经到达终止端点
    fn is_sequence_ended(&self, api_sequence: &ApiSequence) -> bool {
        let functions = &api_sequence.functions;
        let last_fun = functions.last();
        match last_fun {
            None => false,
            Some(api_call) => {
                let (api_type, index) = &api_call.func;
                match api_type {
                    ApiType::BareFunction | ApiType::GenericFunction => {
                        let last_func = &self.api_functions[*index];
                        if last_func._is_end_function(&self.full_name_map) {
                            return true;
                        } else {
                            return false;
                        }
                    }
                    ApiType::ControlStart | ApiType::ScopeEnd | ApiType::UseParam => {
                        // 正常使用不会到达该方法
                        println!("is_sequence_ended wrong");
                        return true;
                    }
                }
            }
        }
    }

    pub fn find_function_by_def_id(&mut self, def_id: LocalDefId) -> Result<&mut ApiFunction, ()>{
        for function in &mut self.api_functions{
            if let ItemId::DefId(def_id_) = function.def_id {
                if def_id.local_def_index == def_id_.index {
                    return Ok(function);
                }
            }
        }
        Err(())
    }

    pub fn get_unsafe_pair(&self, unsafe_indexes: &Vec<usize>,) -> Vec<(usize, usize, Vec<String>)>{
        // TODO:如果unvisited api，需要unsafe struct type的值(move | &mut) 则插入pair中
        
        let mut res_pair = Vec::new();
        if unsafe_indexes.len() < 1 {
            return res_pair;
        }
        for i in 0..unsafe_indexes.len()-1 {
            let index1 = unsafe_indexes[i];
            let struct_types1:Vec<String>;
            let mut struct_types2:Vec<String>;
            if !self.api_functions[index1].rawptr_info.is_empty() {
                struct_types1 = self.api_functions[index1].get_rawptr_struct_types();
            } else if !self.api_functions[index1].drop_info.is_empty() {
                struct_types1 = self.api_functions[index1].get_drop_struct_types();
            } else if !self.api_functions[index1].unsafe_info.is_empty() {
                struct_types1 = self.api_functions[index1].get_all_struct_types();
            } else {
                continue;
            }
        
            for j in i+1..unsafe_indexes.len() {
                let index2 = unsafe_indexes[j];

                if !self.api_functions[index2].rawptr_info.is_empty() {
                    struct_types2 = self.api_functions[index1].get_rawptr_struct_types();
                } else if !self.api_functions[index2].drop_info.is_empty() {
                    struct_types2 = self.api_functions[index2].get_drop_struct_types();
                } else if !self.api_functions[index2].unsafe_info.is_empty() {
                    struct_types2 = self.api_functions[index2].get_all_struct_types();
                } else {
                    continue;
                }

                // println!("{:?}, {:?}", struct_types1, struct_types2);
                let interserct_types = intersect(&struct_types1, &struct_types2);
                if interserct_types.len() > 0{
                    res_pair.push((index1, index2, interserct_types));
                }
            }
        }
        res_pair
    }

    pub fn check_syntax(&self, seq: &ApiSequence) -> bool {
        // println!("Seq: {}", seq);
        let functions = &self.api_functions;
        // 顺序遍历api call查看是否符合语法
        // let mut control_flag = false;
        let mut multi_mut = HashSet::new();
        let mut immutable_borrow = HashSet::new();
        let mut moved_indexes = HashSet::new();
        let mut mutable_borrow_record: Vec<(usize, usize)> = Vec::new();
        let mut immutable_borrow_record: Vec<(usize, usize)> = Vec::new();

        for i in 0..seq.functions.len() {
            let api_call = &seq.functions[i];
            let function = &functions[api_call.func.1];
            let api_type = api_call.func.0;
            match api_type {
                ApiType::BareFunction | ApiType::GenericFunction => {},
                ApiType::ControlStart
                | ApiType::ScopeEnd 
                | ApiType::UseParam => {
                    // control_flag = true;
                    continue;
                },
            }

            for mut_ in &mutable_borrow_record {
                if mut_.1 < i{
                    multi_mut.remove(&mut_.0);
                }
            }
            for immut_ in &immutable_borrow_record {
                if immut_.1 < i {
                    immutable_borrow.remove(&immut_.0);
                }
            }
        
            for j in 0..function.inputs.len() {
                match &api_call.params[j].0 {
                    ParamType::_FuzzableType => {
                        // 还要考虑{}
                        continue;
                    },
                    _ => {}
                }
                let input_ty = &function.inputs[j];
                let index = &api_call.params[j].1;
                let call_type = &api_call.params[j].2;
                // move
                // println!("i:{}, j:{}, index:{}\ninput_ty:{:#?}\ncall_type:{:#?}", i, j, index, input_ty, call_type);
                if api_util::_move_condition(input_ty, call_type) {
                    if multi_mut.contains(&index) || immutable_borrow.contains(&index) || moved_indexes.contains(&index){
                        // println!("value from [{}] borrowed or moved here [{}] after moved", index, i);
                        return false;
                    } else {
                        // 如果该结构没有Copy trait则为move
                        if !self.is_copy_implemented(input_ty) {
                            // println!("api {} move for index {}", i, index);
                            moved_indexes.insert(index);
                        }
                    }
                }

                // borrow
                if api_util::_is_mutable_borrow_occurs(input_ty, call_type) {
                    //如果之前已经被借用过了
                    if multi_mut.contains(&index) || immutable_borrow.contains(&index) || moved_indexes.contains(&index){
                        // println!("mutable borrow error");
                        // println!("value from [{}] mutable borrowed here [{}] after borrowed or moved", index, i);
                        return false;
                    } else {
                        // 查看返回值是否被使用
                        // 一旦该返回值被使用，则在该生命周期之内，该index不能再被借用
                        if let Some(call_index) = seq.func_is_called(i) {
                            // println!("api {} mut borrowed for index {}", i, index);
                            multi_mut.insert(index);
                            mutable_borrow_record.push((*index, call_index));
                        }
                    }
                }

                //如果当前调用是引用，且之前已经被可变引用过，那么这个引用是非法的
                if api_util::_is_immutable_borrow_occurs(input_ty, call_type) {
                    if multi_mut.contains(&index) || moved_indexes.contains(&index) {
                        // println!("value from [{}] immutable borrowed here [{}] after mutable borrowed or moved", index, i);
                        return false;
                    } else {
                        // println!("immutable borrow");
                        // 查看返回值是否被使用
                        // 一旦该返回值被使用，则在该生命周期之内，该index不能再被借用
                        if let Some(call_index) = seq.func_is_called(i) {
                            // println!("api {} immut borrowed for index {}", i, index);
                            immutable_borrow.insert(index);
                            immutable_borrow_record.push((*index, call_index));
                        }
                    }
                }
            }
        }
        // println!("syntax pass!");
        return true;
    }
    pub fn sequence_syntax_analyse(&self, seq: &ApiSequence) -> bool {
        let mut useable_params = vec![false; seq.len()];
        let mut relation_map: HashMap<usize, Vec<(usize, bool)>> = HashMap::new(); // 第n行的返回值和他的输入的关系, true: &mut, false: &
        let mut move_lifetime: HashMap<usize, usize> = HashMap::new();
        let mut mut_ref_lifetime: HashMap<usize, Vec<(usize, usize)>> = HashMap::new(); // 返回值行数，及对应的生命周期（可能有多个）
        let mut immut_ref_lifetime: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        let seq_len = seq.functions.len();
        println!("\n>>>>> SEQUENCE LIFETIME CHECK\n{}", seq);
        for i in 0..seq.functions.len() {
            let api_call = &seq.functions[i];
            let (api_type, func_index) = api_call.func;
            let function = &self.api_functions[func_index];
            let rf_index = match function.return_relate_param() {
                Some(index) => index,
                None => 0,
            };
            // println!("return relate param: {}.{}: {}", seq, i, rf_index);
            match api_type {
                ApiType::ControlStart
                | ApiType::ScopeEnd => {
                    continue;
                },
                _ => {}
            }

            // 使用变量分析
            if let Some(output_ty) = &function.output {
                useable_params[i] = true;
            }
            for j in 0..function.inputs.len() {
                let mut flag = false;
                if rf_index == j + 1 {
                    flag = true;
                }
                if j >= api_call.params.len() {
                    // 这种情况是try_to_call_func时，func只满足部分input时候就进行分析导致的
                    break;
                }
                let input_ty = &function.inputs[j];
                let (param_type, param_index, call_type) = &api_call.params[j];
                if *param_type != ParamType::_FunctionReturn {
                    continue;
                }
                // move分析
                if api_util::_move_condition(input_ty, call_type) {
                    if useable_params[*param_index] {
                        useable_params[*param_index] = false;
                    } else {
                        // 使用已经move过的值
                        println!("SEQUENCE ERROR DETECTED (MOVED {}-{}) <<<<<\n", i, j);
                        return false;
                    }
                    move_lifetime.insert(*param_index, i);
                }
                // mut
                if api_util::_is_mutable_borrow_occurs(input_ty, call_type) {
                    if let Some(mut_lifetimes) = mut_ref_lifetime.get_mut(param_index) {
                        // for (mut_start_index, mut_end_index) in mut_lifetimes {
                        //     if *mut_start_index == i {
                        //         *mut_end_index = i;
                        //     }
                        // }
                        mut_lifetimes.push((i, i));
                    } else {
                        mut_ref_lifetime.insert(*param_index, vec![(i, i)]);
                    }
                    if flag {
                        if let Some(relations) = relation_map.get_mut(&i) {
                            relations.push((*param_index, true));
                        } else {
                            relation_map.insert(i, vec![(*param_index, true)]);
                        }
                    }
                    // 更新lifetime
                    update_lifetime(&relation_map, &mut mut_ref_lifetime, &mut immut_ref_lifetime, *param_index, i);
                }
                // immut
                if api_util::_is_immutable_borrow_occurs(input_ty, call_type) {
                    if let Some(immut_info) = immut_ref_lifetime.get_mut(param_index) {
                        immut_info.push((i, i));
                    } else {
                        immut_ref_lifetime.insert(*param_index, vec![(i, i)]);
                    }
                    if flag {
                        if let Some(relations) = relation_map.get_mut(&i) {
                            relations.push((*param_index, false));
                        } else {
                            relation_map.insert(i, vec![(*param_index, false)]);
                        }
                    }
                    // 更新lifetime
                    update_lifetime(&relation_map, &mut mut_ref_lifetime, &mut immut_ref_lifetime, *param_index, i);
                }
            }
        }
        // 判断原序列是否合法
        if !lifetime_analyse(&move_lifetime, &mut_ref_lifetime, &immut_ref_lifetime) {
            println!("SEQUENCE ERROR DETECTED (LIFETIME) <<<<<\n");
            return false;
        }
        println!("SEQUENCE PASS <<<<<\n");
        return true;
    }
    // 通过move,mut,immut的生命周期持续时间来判断返回值是否可USE
    // 为USE操作服务，找到目前可以使用的所有数据(非基础数据类型)
    // 后续将通过print方法输出（目前未考虑通过调用其他方法）
    // 还可以分析序列的合法性
    pub fn return_usable_analyse(&self, seq: &ApiSequence) -> Option<Vec<bool>> {
        let mut useable_params = vec![false; seq.len()];
        let mut relation_map: HashMap<usize, Vec<(usize, bool)>> = HashMap::new(); // 第n行的返回值和他的输入的关系, true: &mut, false: &
        let mut move_lifetime: HashMap<usize, usize> = HashMap::new();
        let mut mut_ref_lifetime: HashMap<usize, Vec<(usize, usize)>> = HashMap::new(); // 返回值行数，及对应的生命周期（可能有多个）
        let mut immut_ref_lifetime: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        let seq_len = seq.functions.len();
        println!("\n>>>>> SEQUENCE USABLE ANALYSE {}", seq);
        for i in 0..seq.functions.len() {
            let api_call = &seq.functions[i];
            let (api_type, func_index) = api_call.func;
            let function = &self.api_functions[func_index];
            let rf_index = match function.return_relate_param() {
                Some(index) => index,
                None => 0,
            };
            // println!("return relate param: {}.{}: {}", seq, i, rf_index);
            match api_type {
                ApiType::ControlStart
                | ApiType::ScopeEnd => {
                    continue;
                },
                _ => {}
            }
            // 是否moved
            if seq._is_moved(i) {
                // 被moved或者需要drop掉
                continue;
            }
            // 使用变量分析
            if let Some(output_ty) = &function.output {
                useable_params[i] = true;
            }
            for j in 0..function.inputs.len() {
                let mut flag = false;
                if rf_index == j + 1 {
                    flag = true;
                }

                let input_ty = &function.inputs[j];
                println!("{:#?}", api_call);
                let (param_type, param_index, call_type) = &api_call.params[j];
                if *param_type != ParamType::_FunctionReturn {
                    continue;
                }
                // move分析
                if api_util::_move_condition(input_ty, call_type) {
                    if useable_params[*param_index] {
                        useable_params[*param_index] = false;
                    } else {
                        // 使用已经move过的值
                        return None;
                    }
                    move_lifetime.insert(*param_index, i);
                }
                // mut
                if api_util::_is_mutable_borrow_occurs(input_ty, call_type) {
                    if let Some(mut_lifetimes) = mut_ref_lifetime.get_mut(param_index) {
                        // for (mut_start_index, mut_end_index) in mut_lifetimes {
                        //     if *mut_start_index == i {
                        //         *mut_end_index = i;
                        //     }
                        // }
                        mut_lifetimes.push((i, i));
                    } else {
                        mut_ref_lifetime.insert(*param_index, vec![(i, i)]);
                    }
                    if flag {
                        if let Some(relations) = relation_map.get_mut(&i) {
                            relations.push((*param_index, true));
                        } else {
                            relation_map.insert(i, vec![(*param_index, true)]);
                        }
                    }
                    // 更新lifetime
                    update_lifetime(&relation_map, &mut mut_ref_lifetime, &mut immut_ref_lifetime, *param_index, i);
                }
                // immut
                if api_util::_is_immutable_borrow_occurs(input_ty, call_type) {
                    if let Some(immut_info) = immut_ref_lifetime.get_mut(param_index) {
                        immut_info.push((i, i));
                    } else {
                        immut_ref_lifetime.insert(*param_index, vec![(i, i)]);
                    }
                    if flag {
                        if let Some(relations) = relation_map.get_mut(&i) {
                            relations.push((*param_index, false));
                        } else {
                            relation_map.insert(i, vec![(*param_index, false)]);
                        }
                    }
                    // 更新lifetime
                    update_lifetime(&relation_map, &mut mut_ref_lifetime, &mut immut_ref_lifetime, *param_index, i);
                }
            }
        }
        // 判断原序列是否合法
        if !lifetime_analyse(&move_lifetime, &mut_ref_lifetime, &immut_ref_lifetime) {
            println!("SEQUENCE ERROR DETECTED <<<<<\n");
            return None;
        }
        // println!("move lifetime: {:#?}", move_lifetime);
        // println!("mut ref lifetime: {:#?}", mut_ref_lifetime);
        // println!("immut ref lifetime: {:#?}", immut_ref_lifetime);
        // println!("relation map: {:#?}", relation_map);
        // 通过lifetime分析来添加use param
        // 从后往前添加use
        for i in 1..useable_params.len() + 1 {
            let called_index = useable_params.len() - i;
            let call_index = seq_len + i - 1;
            if !useable_params[called_index] {
                continue;
            }
            let immut_ref_lifetime_clone = immut_ref_lifetime.clone();
            // let origin_end_index = immut_ref_lifetime.insert(called_index, call_index);
            if let Some(immut_lifetimes) = immut_ref_lifetime.get_mut(&called_index) {
                immut_lifetimes.push((call_index, call_index));
                // for (mut_start_index, mut_end_index) in mut_lifetimes {
                //     if *mut_start_index == update_index {
                //         *mut_end_index = end_index;
                //         break;
                //     }
                // }
            } else {
                immut_ref_lifetime.insert(called_index, vec![(call_index, call_index)]);
            }
            update_lifetime(&relation_map, &mut mut_ref_lifetime, &mut immut_ref_lifetime, called_index, call_index);
            if !lifetime_analyse(&move_lifetime, &mut_ref_lifetime, &immut_ref_lifetime) {
                println!("SEQUENCE ERROR DETECTED <<<<<\n");
                useable_params[called_index] = false;
                immut_ref_lifetime = immut_ref_lifetime_clone;
            }
        }
        println!("USABLE PARAMS: {:?}", useable_params);
        println!("SEQUENCE USEABLE ANALYSE FINISHED <<<<<");
        Some(useable_params)
    }

    // 用于判断这个结构是否实现了copy
    pub fn is_copy_implemented(&self, ty: &clean::Type) -> bool {
        match ty {
            clean::Type::Path{ path } => {
                if let Some(full_name) = self.full_name_map.get_full_name(&path.def_id()) {
                    if self.copy_structs.contains(&full_name.to_string()) {
                        // println!("copy struct: {}", full_name);
                        return true;
                    }
                    return false; 
                }
                return false;
            },
            _ => {return false;}            
        }
    }

    // 可以直接到达的API（input为0）
    // 或者是纯fuzzable的API
    pub fn is_uncondition_api(&self, index: usize) -> bool {
        let input_params = &self.api_functions[index].inputs;
        if input_params.len() == 0 {
            return true;
        }
        for i in 0..input_params.len() {
            let ty = &input_params[i];
            if !api_util::is_fuzzable_type(ty, &self.full_name_map) {
                return false;
            }
        }
        true
    }

    pub fn set_weight(&mut self) {
        let api_num = self.api_functions.len() as f32;
        let edge_num = self.api_dependencies.len() as f32;
        println!("{} {} {}", api_num, edge_num, ((api_num + edge_num) / 200_f32).tanh());
        let base_weight = 1 + (3_f32 * ((api_num + edge_num) / 200_f32).tanh()) as usize;
        println!("base weight: {}", base_weight);
        
        let mut total_weight = 0;

        for api_func in &mut self.api_functions {
            // 设置初始权重
            api_func.weight += base_weight;
            
            // 如果为unsafe api 则+2
            if api_func.is_unsafe_function() {
                api_func.weight += 2 * base_weight;
            }
            // 检查input params 设置权重
            // 有fuzzable +1
            // 有非fuzzalbe(不可自行生成的变量类型) +1
            let input_params = &api_func.inputs;
            let mut fuzzable_flag = false;
            if input_params.len() != 0 {
                for i in 0..input_params.len() {
                    let ty = &input_params[i];
                    if api_util::is_fuzzable_type(ty, &self.full_name_map) && !fuzzable_flag {
                        api_func.weight += base_weight;
                        fuzzable_flag = true;
                    } else if !api_util::is_fuzzable_type(ty, &self.full_name_map) {
                        api_func.weight += base_weight;
                    }
                }
            }
            //检查返回数据类型
            if let Some(output_param) = &api_func.output {
                if !api_util::is_fuzzable_type(output_param, &self.full_name_map) {
                    api_func.weight += 2 * base_weight;
                }
            }
            total_weight += api_func.weight
        }

        if self.total_weight < total_weight {
            self.total_weight = total_weight;
        }
    }

    pub fn create_graph_dot(&self, covered_nodes: &Vec<usize>, covered_edges: &Vec<(usize, bool)>) {
        let mut content = String::new();
        
        content.push_str("digraph {\n");
        // GraphViz 格式
        content.push_str("\trankdir = TB\n");
        content.push_str("\tsplines = ortho\n");

        // 添加API节点 shape=box
        // 普通API
        // 不安全API color=red
        // 覆盖的普通API style=filled, color=blue
        // 覆盖的不安全API style=filled, color=red
        for i in 0..self.api_functions.len() {
            let api_func = &self.api_functions[i];
            if api_func.is_unsafe_function() {
                if covered_nodes.contains(&i) {
                    content.push_str(
                        // 覆盖的不安全API
                        format!("\tnode{} [label={}, style=filled, color=red]\n", i, i).as_str()
                    );
                } else {
                    content.push_str(
                        // 未覆盖的不安全API
                        format!("\tnode{} [label={}, color=red]\n", i, i).as_str()
                    );
                }
            } else {
                if covered_nodes.contains(&i) {
                    content.push_str(
                        // 覆盖的普通API
                        format!("\tnode{} [label={}, style=filled, color=blue]\n", i, i).as_str()
                    );
                } else {
                    content.push_str(
                        // 未覆盖的普通API
                        format!("\tnode{} [label={}, color=blue]\n", i, i).as_str()
                    );
                }
            }
        }

        // 添加调用边
        // 未覆盖调用边
        // 覆盖调用边 color=red
        for i in 0..self.api_dependencies.len() {
            let output_func = self.api_dependencies[i].0.output_fun.1;
            let input_func = self.api_dependencies[i].0.input_fun.1;
            let param_index = self.api_dependencies[i].0.input_param_index;
            if covered_edges.contains(&(i, false)) || covered_edges.contains(&(i, true)) {
                // 覆盖调用边
                content.push_str(
                    format!("\tnode{} -> node{} [label={}, color=red]\n", output_func, input_func, param_index).as_str()
                );
            } else {
                // 未覆盖调用边
                content.push_str(
                    format!("\tnode{} -> node{} [label={}]\n", output_func, input_func, param_index).as_str()
                );
            }
        }
        
        content.push_str("}");
        let crate_name = self._crate_name.clone();
        let dot_name = file_util::get_dot_path(&crate_name);
        
        if let Ok(mut file) = fs::File::create(dot_name) {
            file.write_all(content.as_bytes()).unwrap();
        }
    }

    pub fn create_api_dependency_graph_visualize(&self) {
        let mut api_content = String::new();
        for func in &self.api_functions {
            api_content.push_str(func.format().as_str());
            api_content.push_str("\n");
        }
        let crate_name = self._crate_name.clone();
        let api_file_path = file_util::get_api_path(&crate_name);
        if let Ok(mut file) = fs::File::create(api_file_path) {
            file.write_all(api_content.as_bytes()).unwrap();
        }

        let dependency_content = self.api_dependency_format();
        let dependency_file_path = file_util::get_dependency_path(&crate_name);
        if let Ok(mut file) = fs::File::create(dependency_file_path) {
            file.write_all(dependency_content.as_bytes()).unwrap();
        }
    }

    pub fn api_dependency_format(&self) -> String {
        let mut dependency_format = String::new();
        for (dependency, unsafe_flag) in &self.api_dependencies {
            
            let unsafe_name = match unsafe_flag {
                true => String::from("[U]"),
                false => String::from("[S]"),
            };
            let input_param_index = dependency.input_param_index;
            // println!("{}: {} in {}-{}", &self.api_functions[dependency.input_fun.1].full_name, input_param_index, &self.api_functions[dependency.input_fun.1].inputs.len(), &self.api_functions[dependency.input_fun.1].input_types.len());
            // api_graph.find_function_by_def_id(*fn_id) 不一定找得到？
            let output_func_name = &self.api_functions[dependency.output_fun.1].full_name;
            let input_func_name = &self.api_functions[dependency.input_fun.1].full_name;
            let call_type_name = dependency.call_type.as_string();
            dependency_format.push_str(format!("{} {} ==[{}]=> {}-{}\n", unsafe_name, output_func_name, call_type_name, input_func_name, dependency.input_param_index).as_str());
        }
        dependency_format
        // [U] fn1 [DirectCall] fn2-0
    }


    // 如果依赖之间是泛型变量，获取所有能够匹配的Type (通过Trait)
    pub fn match_generic_type(
        &self, 
        call_func: &ApiFunction, 
        called_func: &ApiFunction, 
        param_index: usize,
        call_type: &CallType,
    ) -> Option<Vec<ApiParameter>> {
        println!("\n>>> MATCH GENERIC TYPE");
        // println!("Find parameter for: {}->{}-{}", call_func.full_name, called_func.full_name, param_index);
        // 如果不是_GenericCall则直接返回空数组
        if !call_type.is_generic_call() {
            println!("No Generic Call");
            return Some(Vec::new());
        }
        let mut res_parameters: Vec<ApiParameter> = Vec::new();
        let output_type = &call_func.output.as_ref().unwrap();
        let input_type = &called_func.inputs[param_index];
        if api_parameter::is_generic_type(output_type) && api_parameter::is_generic_type(input_type) {
            // GenericType -> GenericType
            println!("GenericType -> GenericType");
            // TraitBound for output:
            let mut output_trait_bounds: Vec<String> = Vec::new();
            if let Some(symbol) = api_parameter::get_generic_name(&output_type) {
                if let Some(trait_bounds) = call_func.get_trait_bound_by_symbol(&symbol) {
                    output_trait_bounds = trait_bounds;
                } else {
                    println!("can't get generic name for output: {}, {}", call_func.full_name, symbol);
                }
            }
            // TraitBound for input:
            let mut input_trait_bounds: Vec<String> = Vec::new();
            if let Some(symbol) = api_parameter::get_generic_name(&input_type) {
                if let Some(trait_bounds) = called_func.get_trait_bound_by_symbol(&symbol) {
                    input_trait_bounds = trait_bounds;
                } else {
                    println!("can't get generic name for input: {}, {}", called_func.full_name, symbol);
                }
            }
            println!("output trait bounds: {:?}", output_trait_bounds);
            println!("input trait bounds: {:?}", input_trait_bounds);
            let union_trait_bounds = output_trait_bounds
                .iter()
                .filter(|&u| !input_trait_bounds.contains(u))
                .chain(&input_trait_bounds)
                .map(|u| u.to_string())
                .collect::<Vec<_>>();

            // 在已有的数据结构里找(包括预设类型)
            for api_param in &self.api_parameters {
                if api_param.is_meet_traits(&union_trait_bounds) {
                    res_parameters.push(api_param.clone());
                }
            }
        } else if api_parameter::is_generic_type(input_type) {
            // SpecialType -> GenericType
            println!("SpecialType -> GenericType");
            let inner_output_type = api_parameter::get_inner_type(output_type);
            // inner_output_type找到对应的api_structure
            if let Some(output_parameter) = self.find_api_parameter_by_clean_type(&inner_output_type) {
                // TraitBound for input:
                let mut input_trait_bounds: Vec<String> = Vec::new();
                if let Some(symbol) = api_parameter::get_generic_name(&input_type) {
                    if let Some(trait_bounds) = called_func.get_trait_bound_by_symbol(&symbol) {
                        input_trait_bounds = trait_bounds;
                    } else {
                        println!("can't get generic name for input: {}, {}", called_func.full_name, symbol);
                    }
                }
                let output_implemented_traits = output_parameter.get_parameter_traits();
                // println!("input_trait_bounds: {:#?}", input_trait_bounds);
                // println!("output_implemented_traits: {:#?}", output_implemented_traits);
                let minusion = input_trait_bounds
                    .iter()
                    .filter(|&u| !output_implemented_traits.contains(u))
                    .collect::<Vec<_>>();
                if minusion.len() == 0 {
                    res_parameters.push(output_parameter);
                }
            }
        } else if api_parameter::is_generic_type(output_type) {
            // GenericType -> SpecialType
            println!("GenericType -> SpecialType");
            let inner_input_type = api_parameter::get_inner_type(input_type);
            // inner_input_type找到对应的api_structure
            if let Some(input_parameter) = self.find_api_parameter_by_clean_type(&inner_input_type) {
                // TraitBound for output:
                let mut output_trait_bounds: Vec<String> = Vec::new();
                if let Some(symbol) = api_parameter::get_generic_name(&output_type) {
                    if let Some(trait_bounds) = call_func.get_trait_bound_by_symbol(&symbol) {
                        output_trait_bounds = trait_bounds;
                    } else {
                        println!("can't get generic name for output: {}, {}", call_func.full_name, symbol);
                    }
                }
                let input_implemented_traits = input_parameter.get_parameter_traits();
                // println!("output_trait_bounds: {:#?}", output_trait_bounds);
                // println!("input_implemented_traits: {:#?}", input_implemented_traits);
                let minusion = output_trait_bounds
                    .iter()
                    .filter(|&u| !input_implemented_traits.contains(u))
                    .collect::<Vec<_>>();
                if minusion.len() == 0 {
                    res_parameters.push(input_parameter);
                }
            } else {
                println!("can't find parameter by clean type: {:#?}", inner_input_type);
            }
        }

        // println!("{:#?}", res_parameters);
        if res_parameters.len() > 0{
            println!("MATCH SUCCESS <<<<<");
            return Some(res_parameters);
        }
        println!("MATCH FAILED <<<<<");
        None
    }

    pub fn find_api_parameter_by_clean_type(&self, clean_type: &clean::Type) -> Option<ApiParameter> {
        // 先判断是不是基础数据类型(预设类型)
        // println!("find api paramater by clean type: {:#?}", clean_type);
        if api_util::is_fuzzable_type(&clean_type, &self.full_name_map) {
            //如果当前参数是fuzzable的
            let fuzzable_call_type =
                fuzzable_type::fuzzable_call_type_by_clean_type(&clean_type, &self.full_name_map);
            let (fuzzable_type, call_type) =
                fuzzable_call_type.generate_fuzzable_type_and_call_type();

            // println!("fuzzable type: {:?}", fuzzable_type);
            // println!("fuzzable call type: {:?}", call_type);
            if let Some(preluded_parameter) = PreludedStructure::new(&fuzzable_type) {
                return Some(ApiParameter::Preluded(preluded_parameter));
            }
        }

        // 在api_graph中的所有结构体中寻找能匹配的
        for api_param in &self.api_parameters {
            // println!("api param: {}", api_param.as_string());
            if api_param.is_same_type(clean_type, &self.full_name_map) {
                // println!("is same type");
                return Some(api_param.clone());
            }
        }
        None
    }

    pub fn find_api_parameter_by_string(&self, param_string: &String) -> Option<ApiParameter> {
        for param in &self.api_parameters {
            if &param.as_string() == param_string {
                return Some(param.clone());
            }
        }
        None
    }

    // 构建Parameter
    pub fn find_structures_related_functions(&mut self) {
        let functions = &self.api_functions.clone();
        for api_param in &mut self.api_parameters {
            for i in 0..functions.len() {
                let func = &functions[i];
                for j in 0..func.inputs.len() {
                    let input_ty = &func.inputs[j];
                    // 泛型
                    if api_parameter::is_generic_type(&input_ty) {
                        let mut input_trait_bounds: Vec<String> = Vec::new();
                        if let Some(symbol) = api_parameter::get_generic_name(&input_ty) {
                            if let Some(trait_bounds) = func.get_trait_bound_by_symbol(&symbol) {
                                input_trait_bounds = trait_bounds;
                            }
                        } else {
                            continue;
                        }
                        let param_traits = api_param.get_parameter_traits();
                        println!("relate input for {}->{}-{}\ntrait bounds: {:#?}\nparam traits:{:#?}", api_param.as_string(), i, j, input_trait_bounds, param_traits);
                        let minusion = input_trait_bounds
                        .iter()
                        .filter(|&u| !param_traits.contains(u))
                        .collect::<Vec<_>>();
                        if minusion.len() == 0 {
                            api_param.add_use_function(i, j);
                        }
                    } else {
                        let inner_input_ty = api_parameter::get_inner_type(&input_ty);
                        if api_param.is_same_type(&inner_input_ty, &self.full_name_map) {
                            api_param.add_use_function(i, j);
                        }
                    }
                }

                if let Some(output_ty) = &func.output {
                    // 泛型
                    if api_parameter::is_generic_type(&output_ty) {
                        let mut output_trait_bounds: Vec<String> = Vec::new();
                        if let Some(symbol) = api_parameter::get_generic_name(&output_ty) {
                            if let Some(trait_bounds) = func.get_trait_bound_by_symbol(&symbol) {
                                output_trait_bounds = trait_bounds;
                            }
                        } else {
                            continue;
                        }
                        let param_traits = api_param.get_parameter_traits();
                        println!("relate output for {}->{}\ntrait bounds: {:#?}\nparam traits:{:#?}", i, api_param.as_string(), output_trait_bounds, param_traits);
                        let minusion = output_trait_bounds
                        .iter()
                        .filter(|&u| !param_traits.contains(u))
                        .collect::<Vec<_>>();
                        if minusion.len() == 0 {
                            api_param.add_return_function(i);
                        }
                    } else {
                        // 确定类型
                        let inner_output_ty = api_parameter::get_inner_type(&output_ty);
                        if api_param.is_same_type(&inner_output_ty, &self.full_name_map) {
                            api_param.add_return_function(i);
                        }
                    }
                }
            }
        }
    }

    // 初始化预设数据结构类型
    // 目前有:
    // String
    // PanicStruct(会导致pannic的structure)
    pub fn init_preluded_structure(&mut self) {
        // String
        if let Some(preluded_string) = PreludedStructure::new(&FuzzableType::String) {
            self.api_parameters.push(ApiParameter::Preluded(preluded_string));
        }
    }

    pub fn is_preluded_param(&self, name: &str) -> Option<ApiParameter> {
        for api_param in &self.api_parameters {
            if api_param.as_string() == format!("Preluded({})", name) {
                return Some(api_param.clone());
            }
        }
        None
    }

    // 检查是否序列中的泛型变量都在generic_map中有对应的parameter
    pub fn check_generic_refered(&self, seq: &ApiSequence) -> bool {
        for api_call in &seq.functions {
            let function = &self.api_functions[api_call.func.1];
            let param_len = match function.output {
                Some(_) => function.inputs.len() + 1,
                None => function.inputs.len(),
            };
            for i in 0..param_len {
                let mut generic_path = String::new();
                if i == param_len - 1 {
                    if let Some(generic_path_) = function.get_generic_path_for_output() {
                        generic_path = generic_path_;
                    }
                } else {
                    if let Some(generic_path_) = function.get_generic_path_by_param_index(i) {
                        generic_path = generic_path_;
                    }
                }
                if generic_path == String::new() {
                    continue;
                }
                match seq.get_generic_parameter(&generic_path) {
                    Some(_) => { continue; }
                    None => { 
                        println!("{:#?}", seq);
                        return false; 
                    }
                }
            }
        }
        true
    }

    // 获取edge数量
    pub fn get_total_edge(&self) -> usize {
        let mut dependency_pair = Vec::new();
        for (dependency, unsafe_flag) in &self.api_dependencies {
            let output_func_index = dependency.output_fun.1;
            let input_func_index = dependency.input_fun.1;
            if !dependency_pair.contains(&(output_func_index, input_func_index)) {
                dependency_pair.push((output_func_index, input_func_index));
            }
        }
        dependency_pair.len()
    }

    // 获取edge数量
    pub fn get_total_unsafe_edge(&self) -> usize {
        let mut dependency_pair = Vec::new();
        for (dependency, unsafe_flag) in &self.api_dependencies {
            let output_func_index = dependency.output_fun.1;
            let input_func_index = dependency.input_fun.1;
            if *unsafe_flag && !dependency_pair.contains(&(output_func_index, input_func_index)) {
                dependency_pair.push((output_func_index, input_func_index));
            }
        }
        dependency_pair.len()
    }
}

/**
 * 两个数组相交
 * 并过滤基本数据类型的变量类型
 */
pub fn intersect(nums1: &Vec<String>, nums2: &Vec<String>) -> Vec<String> {
   let mut map:HashMap<String, i32> = HashMap::new();
   let mut vec:Vec<String> = Vec::new();
   let primitive_type:Vec<String> = ["i8","i16","i32","i64","i128","isize","u8","u16","u32","u64","u128","usize","char","str","f32","f64","bool"]
                                    .into_iter()
                                    .map(|value| value.to_string())
                                    .collect();
   for e in nums1{
       if map.contains_key(e){
           map.insert(e.clone(), map[e]+1);
       }else{
           map.insert(e.clone(), 1);
       }
   }
   for e in nums2{
       if map.contains_key(e){         
          if  map[e] > 0 {
            if !primitive_type.contains(e){  
                vec.push(e.clone())
            }
          }
          map.insert(e.clone(),map[e]-1);
       }
   }
   return vec;
}


/**
 * 把复杂数据类型还原成最初始的样子
 */
pub fn get_naked_type(struct_type: &String) -> String {
    let mut type_ = struct_type.clone();
    if type_.starts_with("&mut") {
        type_ = type_.chars().skip(5).collect();
    } else if type_.starts_with('&') {
        type_ = type_.chars().skip(1).collect();
    } else if type_.starts_with("*const") {
        type_ = type_.chars().skip(7).collect();
    } else if type_.starts_with("*mut") {
        type_ = type_.chars().skip(5).collect();
    } else if type_.starts_with('*') {
        type_ = type_.chars().skip(1).collect();
    }
    type_
}