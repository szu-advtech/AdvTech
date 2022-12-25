use crate::fuzz_target::afl_util::{self, _AflHelpers};
use crate::fuzz_target::api_graph::{ApiGraph, ApiType};
use crate::fuzz_target::api_function::ApiFunction;
use crate::fuzz_target::api_util;
use crate::fuzz_target::call_type::CallType;
use crate::fuzz_target::fuzzable_type::FuzzableType;
use crate::fuzz_target::prelude_type;
use crate::fuzz_target::replay_util;
use crate::fuzz_target::api_parameter::{ApiStructure, ApiParameter, GenericPath};
use std::collections::{HashMap, HashSet};
use std::fmt;
use rustdoc::clean;

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum ParamType {
    _FunctionReturn,
    _FuzzableType,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct ApiCall {
    pub func: (ApiType, usize), //要调用的函数类型，以及在对应数组中的位置
    pub params: Vec<(ParamType, usize, CallType)>, //参数类型(表示是使用之前的返回值，还是使用fuzzable的变量)，在当前的调用序列中参数所在的位置，以及如何调用
    pub is_unsafe: bool,
    pub is_get_raw_ptr: bool,
    pub is_drop: bool,
    pub is_mutate: bool,
    pub output_type: Option<clean::Type>,
}

impl ApiCall {
    pub fn new_without_params(api_type: &ApiType, index: usize) -> Self {
        let func = (api_type.clone(), index);
        let params = Vec::new();
        let is_unsafe = false;
        let is_get_raw_ptr = false;
        let is_drop = false;
        let is_mutate = false;
        let output_type = None;
        ApiCall { func, params, is_unsafe, is_get_raw_ptr, is_mutate, is_drop, output_type }
    }

    pub fn new(fun_index: usize, output: Option<clean::Type>, is_generic: bool) -> Self {
        let api_type = match is_generic {
            true => ApiType::GenericFunction,
            false => ApiType::BareFunction,          
        };
        let func = (api_type, fun_index);
        let params = Vec::new();
        let is_unsafe = false;
        let is_get_raw_ptr = false;
        let is_drop = false;
        let is_mutate = false;
        let output_type = output;
        ApiCall { func, params, is_unsafe, is_get_raw_ptr, is_drop, is_mutate, output_type }
    }

    pub fn new_control_start(param_index: usize) -> Self{
        let api_type = ApiType::ControlStart;
        let func = (api_type, 0);
        let mut params = Vec::new();
        let param = (ParamType::_FuzzableType, param_index, CallType::_DirectCall);
        params.push(param);
        let is_unsafe = false;
        let is_get_raw_ptr = false;
        let is_drop = false;
        let is_mutate = false;
        let output_type = None;
        ApiCall { func, params, is_unsafe, is_get_raw_ptr, is_drop, is_mutate, output_type}
    }

    pub fn new_scope_end() -> Self{
        let api_type = ApiType::ScopeEnd;
        let func = (api_type, 0);
        let params = Vec::new();
        let is_unsafe = false;
        let is_get_raw_ptr = false;
        let is_drop = false;
        let is_mutate = false;
        let output_type = None;
        ApiCall { func, params, is_unsafe, is_get_raw_ptr, is_drop, is_mutate, output_type}
    }

    pub fn _add_param(&mut self, param_type: ParamType, param_index: usize, call_type: CallType) {
        self.params.push((param_type, param_index, call_type));
    }

    pub fn set_output_type(&mut self, output_ty:Option<clean::Type>) {
        self.output_type = output_ty;
    }
}

//function call sequences
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ApiSequence {
    //TODO:如何表示函数调用序列？
    pub functions: Vec<ApiCall>,               //函数调用序列
    pub fuzzable_params: Vec<FuzzableType>,    //需要传入的fuzzable变量
    pub _using_traits: Vec<String>,            //需要use引入的traits的路径
    pub _unsafe_tag: bool,                     //标志这个调用序列是否需要加上unsafe标记
    pub _moved: HashSet<usize>,                //表示哪些返回值已经被move掉，不再能被使用
    pub _fuzzable_mut_tag: HashSet<usize>,     //表示哪些fuzzable的变量需要带上mut标记
    pub _function_mut_tag: HashSet<usize>,     //表示哪些function的返回值需要带上mut标记
    pub _covered_dependencies: HashSet<(usize, bool)>, //表示用到了哪些dependency,即边覆盖率,bool判断该边是否到达unsafe API
    pub _control_nums: usize,                  //表示有几个控制块
    pub _pattern_mark: Option<String>,         //用于标注模式类型，如果没有pattern则为None
    pub generic_map: HashMap<GenericPath, ApiParameter> //queue::Queue::T -> String
}

impl ApiSequence {
    pub fn new() -> Self {
        let functions = Vec::new();
        let fuzzable_params = Vec::new();
        let _using_traits = Vec::new();
        let _unsafe_tag = false;
        let _moved = HashSet::new();
        let _fuzzable_mut_tag = HashSet::new();
        let _function_mut_tag = HashSet::new();
        let _covered_dependencies = HashSet::new();
        let _control_nums = 0;
        let _pattern_mark = None;
        let generic_map = HashMap::new();
        ApiSequence {
            functions,
            fuzzable_params,
            _using_traits,
            _unsafe_tag,
            _moved,
            _fuzzable_mut_tag,
            _function_mut_tag,
            _covered_dependencies,
            _control_nums,
            _pattern_mark,
            generic_map,
        }
    }

    /**
     * 删除序列中最后一个API所对应的input_index的struct生成相关的API序列
     * 删除完的序列是非法的，需要重新指向才可以合法。
     * 插入control的(if{})的序列时 shift为1
     */
    pub fn remove_input_create_api(&self, input_index: usize, shift: usize) -> Option<Self>{
        let mut res_seq = self.clone();
        let seq_len = &res_seq.functions.len();

        let last_api_call = &res_seq.functions[seq_len-1-shift];
        let start_index = match last_api_call.params[input_index].0{
            ParamType::_FunctionReturn => {
                last_api_call.params[input_index].1
            }
            _ => {
                return None;
            }
        };
        
        // 获取删除的API序号
        let mut delete_api = Vec::new();
        delete_api.push(start_index);
        let mut next_indexs = Vec::new();
        next_indexs.push(start_index);
        let mut next_index;
        loop {
            if next_indexs.len() == 0 {break;}
            next_index = next_indexs.pop().unwrap();
            for param in &res_seq.functions[next_index].params{
                match param.0 {
                    ParamType::_FunctionReturn => {
                        next_indexs.push(param.1);
                        delete_api.push(param.1);
                    },
                    ParamType::_FuzzableType => {
                        // TODO: 删除fuzzable type的指向关系
                        // if start_index != next_index {
                        //     res_seq.fuzzable_params.remove(param.1);
                        // }
                    },
                    // _ => {}
                }
            }
        }

        // 删除API，并维持链接关系
        // 由大到小排序delete_api
        delete_api.sort();
        delete_api.reverse();
        // println!("Delete API: {:#?}", delete_api);
        let mut i = seq_len - 1;
        loop {
            for j in &delete_api {
                if i == *j {
                    res_seq.functions.remove(*j);
                    break;
                } else if i > *j {
                    let api_call = &mut res_seq.functions[i];
                    for input_param in &mut api_call.params {
                        match input_param.0 {
                            ParamType::_FunctionReturn => {
                                if input_param.1 > *j {
                                    input_param.1 -= 1;
                                }
                            },
                            ParamType::_FuzzableType => {}
                        }
                    }
                    continue;
                }
            }
            if i == 0{
                break;
            }
            i -= 1;
        }
        return Some(res_seq);
    }

    pub fn remove_duplicate_param(&mut self) {
        // println!("Before: {}", self);
        let mut func_record = HashSet::new();
        let mut func_replace = Vec::new(); // 元组(i, j)代表:删除i,保留j
        let mut func_map = HashMap::new();
        // 记录替换的方法(保留先出现的)
        for i in 0..self.functions.len() {
            let api_call = &mut self.functions[i];
            let func_index = api_call.func.1;
            if !func_record.insert(func_index) {
                func_replace.push((i, *func_map.get(&func_index).unwrap()));
            } else{
                func_map.insert(func_index, i);
            }
        }

        // 翻转func_replace,避免后面遍历导致问题(delete func 由小变大，但是新的指向又大变小，经过if判断导致需要减小的指向没有减小)
        func_replace.reverse();
        // println!("{:?}", func_replace);
        // 从后往前删除api_call
        let mut i = self.functions.len()-1;
        while i > 0 {
            for (j, k) in &func_replace{
                if i == *j {
                    // 删除该api call
                    self.functions.remove(*j);
                    break;
                } else if i > *j {
                    // i > j的情况，指向需要减1
                    let api_call = &mut self.functions[i];
                    for input_param in &mut api_call.params {
                        match input_param.0 {
                            ParamType::_FunctionReturn => {
                                if input_param.1 > *j {
                                    input_param.1 -= 1;
                                } else if input_param.1 == *j {
                                    input_param.1 = *k;
                                }
                            },
                            ParamType::_FuzzableType => {}
                        }
                    }
                    continue;
                }
            }
            i -= 1;
        }
        // println!("After: {}", self);
    }

    pub fn is_contain(&self, other_seq: &ApiSequence) -> bool {
        if self.functions.len() < other_seq.functions.len() {
            return false;
        }
        if self.functions.len() == other_seq.functions.len() {
            return self.get_sequence_string() == other_seq.get_sequence_string();
        }
        // 判断是否包含关系
        // 如 12345 和 135,125等
        // 逻辑：遍历短的序列(s2)，看每一个char是否能在s1上找到
        // 如果能，得到该index，并将s1变成index之后的String
        // 重复直到
        let mut s1 = self.get_sequence_string();
        let s2 = other_seq.get_sequence_string();
        let mut seq_chars = s2.chars();
        loop {
            match seq_chars.next() {
                Some(value) => {
                    match s1.find(value) {
                        Some(pos) => {
                            s1 = s1.get(pos+1..).unwrap_or("").to_string();
                        }
                        None => return false,
                    }
                }
                None => return true,
            }
        }
    }

    pub fn get_sequence_string(&self) -> String {
        let mut s = String::new();
        for i in 0..self.functions.len() {
            let f = &self.functions[i];
            s += &f.func.1.to_string();
            if i < self.functions.len()-1 {
                s += "->";
            }
        }
        // s += format!("{:#?}", self.generic_map);
        for (generic_path, api_parameter) in &self.generic_map {
            s += " ";
            s += &generic_path.as_string();
            s += ":";
            s += &api_parameter.as_string();
        }
        s
    }

    pub fn _add_fn_without_params(&mut self, api_type: &ApiType, index: usize) {
        let api_call = ApiCall::new_without_params(api_type, index);
        self.functions.push(api_call);
    }

    pub fn _add_dependency(&mut self, dependency: (usize, bool)) {
        self._covered_dependencies.insert(dependency);
    }

    pub fn len(&self) -> usize {
        self.functions.len()
    }

    pub fn _has_no_fuzzables(&self) -> bool {
        if self.fuzzable_params.len() <= 0 {
            return true;
        } else {
            return false;
        }
    }

    pub fn _last_api_func_index(&self) -> Option<usize> {
        if self.len() != 0 {
            for i in 0..self.len(){
                let last_api_call = &self.functions[self.len() - i - 1];
                let (api_type, index) = last_api_call.func;
                match api_type {
                    ApiType::BareFunction | ApiType::GenericFunction => { return Some(index) },
                    ApiType::ControlStart | ApiType::ScopeEnd | ApiType::UseParam => continue,
                }
            }
        }
        None
    }

    /**
     * 用于merged_seq中插入unvisited api seq
     * other_seq: unvisited api sequence
     * target_index: insert location
     * create_index: unsafe create location
     * delete_index: unsafe struct input index
     * unvisited_index: unvisited function indexs
     */
    pub fn _insert_another_sequence(
        &mut self, 
        other_seq: &ApiSequence, 
        target_index: usize, 
        create_index: usize, 
        delete_index: usize,
        unvisited_index: usize,
        insert_call_type: &CallType,
    ) {
        // 目前是针对unsafe的insert，会关联unsafe struct type
        let first_func_number = self.functions.len();
        let first_fuzzable_number = self.fuzzable_params.len();
        let mut insert_index = target_index;
        // println!("other sequence {:#?}\ndelete idex {}", other_seq, delete_index);
        // 删减other_seq的unsafe 生成部分
        if let Some(mut other_sequence) = other_seq.remove_input_create_api(delete_index, 0){
            // println!("other sequence remove input create api {}", other_sequence);
            // 例如对于：1 2 3 4序列
            // 1 2 ^ 3 4 若target_index=2，则在^中进行插入
            
            // 处理1 2 ^ 部分
            // Unsafe 部分需要同源化
            for other_function in &other_sequence.functions {
                let other_func = other_function.func.clone();
                let mut new_other_params = Vec::new();
                for i in 0..other_function.params.len(){
                    let (api_type, func_index) = &other_function.func;
                    let (param_type, index, call_type) = &other_function.params[i];
                    if i == delete_index && *func_index == unvisited_index {
                        let new_index = match param_type {
                            ParamType::_FuzzableType => create_index,
                            ParamType::_FunctionReturn => create_index,
                        };
                        new_other_params.push((param_type.clone(), new_index, insert_call_type.clone()));
                    } else {
                        let new_index = match param_type {
                            ParamType::_FuzzableType => *index + first_fuzzable_number,
                            ParamType::_FunctionReturn => *index + target_index,
                        };
                        new_other_params.push((param_type.clone(), new_index, call_type.clone()));
                    }
                }
                let new_other_function = ApiCall {
                    func: other_func,
                    params: new_other_params,
                    is_unsafe: other_function.is_unsafe,
                    is_get_raw_ptr: other_function.is_get_raw_ptr,
                    is_drop: other_function.is_drop,
                    is_mutate: other_function.is_mutate,
                    output_type: other_function.output_type.clone(),
                };
                self.functions.insert(insert_index, new_other_function);
                insert_index += 1;
            }

            // 更新3 4部分
            // 3 4指向大于insert_index的部分会受到影响
            let shift = insert_index - target_index;
            for i in insert_index..self.functions.len(){
                for param in &mut self.functions[i].params{
                    match param.0{
                        ParamType::_FuzzableType => {},
                        ParamType::_FunctionReturn => {
                            if param.1 >= target_index {
                                param.1 += shift;
                            }
                        }
                    }
                }
            }

            //更新fuzzable_params
            self.fuzzable_params.append(&mut other_sequence.fuzzable_params);
            //using_trait
            self._using_traits.append(&mut other_sequence._using_traits);
        }
    }

    pub fn _merge_another_sequence(&self, other: &ApiSequence) -> Self {
        let mut res = self.clone();
        let first_func_number = res.functions.len();
        let first_fuzzable_number = res.fuzzable_params.len();
        let mut other_sequence = other.clone();
        //functions
        for other_function in &other_sequence.functions {
            let other_func = other_function.func.clone();
            let mut new_other_params = Vec::new();
            for (param_type, index, call_type) in &other_function.params {
                let new_index = match param_type {
                    ParamType::_FuzzableType => *index + first_fuzzable_number,
                    ParamType::_FunctionReturn => *index + first_func_number,
                };
                new_other_params.push((param_type.clone(), new_index, call_type.clone()));
            }
            let new_other_function = ApiCall {
                func: other_func,
                params: new_other_params,
                is_unsafe: other_function.is_unsafe,
                is_get_raw_ptr: other_function.is_get_raw_ptr,
                is_drop: other_function.is_drop,
                is_mutate: other_function.is_mutate,
                output_type: other_function.output_type.clone(),
            };
            res.functions.push(new_other_function);
        }
        //fuzzable_params
        res.fuzzable_params
            .append(&mut other_sequence.fuzzable_params);
        //using_trait
        res._using_traits.append(&mut other_sequence._using_traits);
        //unsafe tag
        res._unsafe_tag = if other_sequence._unsafe_tag {
            other_sequence._unsafe_tag
        } else {
            res._unsafe_tag
        };
        //move tag
        for move_tag in other_sequence._moved {
            res._moved.insert(move_tag + first_func_number);
        }
        //fuzzable mut tag
        for fuzzable_mut_tag in other_sequence._fuzzable_mut_tag {
            res._fuzzable_mut_tag
                .insert(fuzzable_mut_tag + first_fuzzable_number);
        }
        //function mut tag
        for function_mut_tag in other_sequence._function_mut_tag {
            res._function_mut_tag
                .insert(function_mut_tag + first_func_number);
        }
        // generic map
        for (generic_name1, param1) in &other.generic_map {
            let mut add_flag = true;
            for (generic_name2, param2) in &self.generic_map {
                if generic_name1 == generic_name2 {
                    // 都有该generic name
                    if param1 != param2 {
                        // 如果paramter不一样
                        // 先保持param2的，即不更改
                        panic!("merge param1 != param2")
                    }
                    add_flag = false;
                    break;
                }
            }
            // 如果add_flag为真，即generic_name1在generic_name2中没有
            if add_flag {
                res.add_generic_info(&generic_name1.full_name, param1.clone());
            }
        }
        res
    }

    pub fn _merge_sequences(sequences: &Vec<ApiSequence>) -> Self {
        let sequences_len = sequences.len();
        if sequences_len <= 0 {
            //println!("Should not merge with no sequence");
            return ApiSequence::new();
        }
        let mut basic_sequence = sequences.first().unwrap().clone();
        for i in 1..sequences_len {
            let other_sequence = &sequences[i];
            basic_sequence = basic_sequence._merge_another_sequence(other_sequence);
        }
        basic_sequence
    }

    pub fn _form_control_block(&mut self){
        // 在开头添加ControlStart
        // 在结尾添加ScopeEnd
        let fuzzable_num = self.fuzzable_params.len();
        let control_start = ApiCall::new_control_start(fuzzable_num);
        let control_end = ApiCall::new_scope_end();
        for func in &mut self.functions{
            for param in &mut func.params{
                match param.0 {
                    ParamType::_FuzzableType => {}
                    ParamType::_FunctionReturn => {
                        param.1 += 1;
                    }
                }
            }
        }
        // 暂时用bool变量随机吧
        use rustdoc::clean::PrimitiveType;
        self.fuzzable_params.push(FuzzableType::Primitive(PrimitiveType::Bool));
        self.functions.insert(0, control_start);
        self.functions.push(control_end);
    }

    pub fn _contains_api_function(&self, index: usize) -> bool {
        for api_call in &self.functions {
            let (_, func_index) = api_call.func;
            if index == func_index {
                return true;
            }
        }
        return false;
    }

    pub fn _get_contained_api_functions(&self) -> Vec<usize> {
        let mut res = Vec::new();
        for api_call in &self.functions {
            let (_, func_index) = &api_call.func;
            if !res.contains(func_index) {
                res.push(*func_index);
            }
        }
        res
    }

    pub fn _get_contained_unsafe_api_functions(&self, api_graph: &ApiGraph) -> Vec<usize> {
        let mut res = Vec::new();
        for api_call in &self.functions {
            let (_, func_index) = &api_call.func;
            let function = &api_graph.api_functions[*func_index];
            if !res.contains(func_index) && function.is_unsafe_function() {
                res.push(*func_index);
            }
        }
        res
    }

    pub fn _is_moved(&self, index: usize) -> bool {
        if self._moved.contains(&index) {
            true
        } else {
            false
        }
    }

    pub fn _insert_move_index(&mut self, index: usize) {
        self._moved.insert(index);
    }

    pub fn _add_fn(&mut self, api_call: ApiCall) {
        self.functions.push(api_call);
    }

    pub fn _insert_fuzzable_mut_tag(&mut self, index: usize) {
        self._fuzzable_mut_tag.insert(index);
    }

    pub fn _is_fuzzable_need_mut_tag(&self, index: usize) -> bool {
        if self._fuzzable_mut_tag.contains(&index) {
            true
        } else {
            false
        }
    }

    pub fn _insert_function_mut_tag(&mut self, index: usize) {
        self._function_mut_tag.insert(index);
    }

    pub fn _is_function_need_mut_tag(&self, index: usize) -> bool {
        if self._function_mut_tag.contains(&index) {
            true
        } else {
            false
        }
    }

    pub fn set_unsafe(&mut self) {
        self._unsafe_tag = true;
    }

    pub fn add_trait(&mut self, trait_full_path: &String) {
        self._using_traits.push(trait_full_path.clone());
    }

    pub fn _is_fuzzables_fixed_length(&self) -> bool {
        for fuzzable_param in &self.fuzzable_params {
            if !fuzzable_param._is_fixed_length() {
                return false;
            }
        }
        return true;
    }

    pub fn _fuzzables_min_length(&self) -> usize {
        let mut total_length = 0;
        for fuzzable_param in &self.fuzzable_params {
            total_length = total_length + fuzzable_param._min_length();
        }
        total_length
    }

    pub fn _contains_multi_dynamic_length_fuzzable(&self) -> bool {
        for fuzzable_param in &self.fuzzable_params {
            if fuzzable_param._is_multiple_dynamic_length() {
                return true;
            }
        }
        false
    }

    pub fn _fuzzable_fixed_part_length(&self) -> usize {
        let mut total_length = 0;
        for fuzzable_param in &self.fuzzable_params {
            total_length = total_length + fuzzable_param._fixed_part_length();
        }
        total_length
    }

    pub fn _dynamic_length_param_number(&self) -> usize {
        let mut total_number = 0;
        for fuzzable_param in &self.fuzzable_params {
            total_number = total_number + fuzzable_param._dynamic_length_param_number();
        }
        total_number
    }

    pub fn _dead_code(&self, _api_graph: &ApiGraph) -> Vec<bool> {
        let sequence_len = self.len();
        let mut dead_api_call = Vec::new();
        for _ in 0..sequence_len {
            dead_api_call.push(true);
        }

        let mut used_params = HashMap::new(); // param_index, 最后一次使用这个param的api call index

        let api_call_num = self.functions.len();
        for api_call_index in 0..api_call_num {
            let api_call = &self.functions[api_call_index];
            let params = &api_call.params;
            for (param_type, index, _) in params {
                if let ParamType::_FunctionReturn = param_type {
                    dead_api_call[*index] = false;
                    used_params.insert(*index, api_call_index);
                }
                if let ParamType::_FuzzableType = param_type {
                    if let ApiType::UseParam = api_call.func.0 {
                        dead_api_call[*index] = false;
                        used_params.insert(*index, api_call_index);
                    }
                }
            }
        }

        for api_call_index in 0..api_call_num {
            if !dead_api_call[api_call_index] {
                continue;
            }
            let api_call = &self.functions[api_call_index];
            let params = &api_call.params;
            let param_len = params.len();
            if param_len <= 0 {
                continue;
            }
            let api_function_index = match api_call.func.0 {
                ApiType::BareFunction | ApiType::GenericFunction => api_call.func.1,
                ApiType::ControlStart | ApiType::ScopeEnd | ApiType::UseParam => continue,
            };
            let api_function = &_api_graph.api_functions[api_function_index];
            for param_index in 0..param_len {
                if param_index >= api_function.inputs.len() {
                    println!("Error Seq: {:#?}", self);
                }
                let input_type = &api_function.inputs[param_index];
                let (param_type, index, call_type) = &params[param_index];
                if let ParamType::_FunctionReturn = *param_type {
                    if api_util::_is_mutable_borrow_occurs(input_type, call_type) {
                        if let Some(last_call_index) = used_params.get(index) {
                            if api_call_index < *last_call_index {
                                dead_api_call[api_call_index] = false;
                            }
                        }
                    }
                }
            }
        }

        dead_api_call
    }

    // modify logic
    // 原本：have dead_codes -> true
    // 改成：all dead_codes -> true
    pub fn _contains_dead_code_except_last_one(&self, _api_graph: &ApiGraph) -> bool {
        // println!("dead code seq: {}", self);
        let sequence_len = self.len();
        if sequence_len <= 1 {
            return false;
        }
        let dead_codes = self._dead_code(_api_graph);
        for i in 0..sequence_len - 1 {
            if dead_codes[i] {
                return true;
            }
        }
        return false;
    }

    pub fn _to_replay_crash_file(&self, _api_graph: &ApiGraph, test_index: usize) -> String {
        let mut res = self._to_afl_except_main(_api_graph, test_index);
        res = res.replace("#[macro_use]\nextern crate afl;\n", "");
        res.push_str(replay_util::_read_crash_file_data());
        res.push('\n');
        res.push_str(self._reproduce_main_function(test_index).as_str());
        res
    }

    pub fn _to_afl_test_file(&self, _api_graph: &ApiGraph, test_index: usize) -> String {
        let mut res = self._to_afl_except_main(_api_graph, test_index);
        res.push_str(self._afl_main_function(test_index).as_str());
        res
    }

    pub fn _to_libfuzzer_test_file(&self, _api_graph: &ApiGraph, test_index: usize) -> String {
        let mut res = self._to_afl_except_main(_api_graph, test_index);
        res = res.replace(
            "#[macro_use]\nextern crate afl;\n",
            format!("#![no_main]\n#[macro_use]\nextern crate libfuzzer_sys;\n").as_str(),
        );
        res.push_str(self._libfuzzer_fuzz_main(test_index).as_str());
        res
    }

    pub fn _libfuzzer_fuzz_main(&self, test_index: usize) -> String {
        let mut res = String::new();
        res.push_str("fuzz_target!(|data: &[u8]| {\n");
        res.push_str(self._afl_closure_body(0, test_index).as_str());
        res.push_str("});\n");
        res
    }

    pub fn _to_afl_except_main(&self, _api_graph: &ApiGraph, test_index: usize) -> String {
        let mut res = String::new();
        //加入可能需要开启的feature gate
        let feature_gates = afl_util::_get_feature_gates_of_sequence(&self.fuzzable_params);

        if feature_gates.is_some() {
            for feature_gate in &feature_gates.unwrap() {
                let feature_gate_line = format!("{feature_gate}\n", feature_gate = feature_gate);
                res.push_str(feature_gate_line.as_str());
            }
        }

        res.push_str("#[macro_use]\n");
        res.push_str("extern crate afl;\n");
        res.push_str(format!("extern crate {};\n", _api_graph._crate_name).as_str());

        let prelude_helper_functions = self._prelude_helper_functions();
        if let Some(prelude_functions) = prelude_helper_functions {
            res.push_str(prelude_functions.as_str());
        }

        let afl_helper_functions = self._afl_helper_functions();
        if let Some(afl_functions) = afl_helper_functions {
            res.push_str(afl_functions.as_str());
        }
        res.push_str(
            self._to_well_written_function(_api_graph, test_index, 0)
                .as_str(),
        );
        res.push('\n');
        res
    }

    pub fn _prelude_helper_functions(&self) -> Option<String> {
        let mut prelude_helpers = HashSet::new();
        for api_call in &self.functions {
            let params = &api_call.params;
            for (_, _, call_type) in params {
                let helpers = prelude_type::_PreludeHelper::_from_call_type(call_type);
                for helper in helpers {
                    prelude_helpers.insert(helper);
                }
            }
        }
        if prelude_helpers.len() == 0 {
            return None;
        }
        let mut res = String::new();
        for helper in prelude_helpers {
            res.push_str(helper._to_helper_function());
            res.push('\n');
        }
        Some(res)
    }

    pub fn _afl_helper_functions(&self) -> Option<String> {
        let afl_helper_functions =
            afl_util::_get_afl_helpers_functions_of_sequence(&self.fuzzable_params);
        match afl_helper_functions {
            None => None,
            Some(afl_helpers) => {
                let mut res = String::new();
                for afl_helper in &afl_helpers {
                    res.push_str(format!("{}\n", afl_helper).as_str());
                }
                Some(res)
            }
        }
    }

    pub fn _afl_main_function(&self, test_index: usize) -> String {
        let mut res = String::new();
        let indent = _generate_indent(4);
        res.push_str("fn main() {\n");
        res.push_str(indent.as_str());
        res.push_str("fuzz!(|data: &[u8]| {\n");
        res.push_str(self._afl_closure_body(4, test_index).as_str());
        res.push_str(indent.as_str());
        res.push_str("});\n");
        res.push_str("}\n");
        res
    }

    pub fn _reproduce_main_function(&self, test_index: usize) -> String {
        format!(
            "fn main() {{
    let _content = _read_data();
    let data = &_content;
    println!(\"data = {{:?}}\", data);
    println!(\"data len = {{:?}}\", data.len());
{}
}}",
            self._afl_closure_body(0, test_index)
        )
    }

    pub fn _afl_closure_body(&self, outer_indent: usize, test_index: usize) -> String {
        let extra_indent = 4;
        let mut res = String::new();
        let indent = _generate_indent(outer_indent + extra_indent);
        res.push_str(format!("{indent}//actual body emit\n", indent = indent).as_str());

        let op = if self._is_fuzzables_fixed_length() {
            "!="
        } else {
            "<"
        };
        let min_len = self._fuzzables_min_length();
        res.push_str(
            format!(
                "{indent}if data.len() {op} {min_len} {{return;}}\n",
                indent = indent,
                op = op,
                min_len = min_len
            )
            .as_str(),
        );

        let dynamic_param_start_index = self._fuzzable_fixed_part_length();
        let dynamic_param_number = self._dynamic_length_param_number();
        let dynamic_length_name = "dynamic_length";
        let every_dynamic_length = format!("let {dynamic_length_name} = (data.len() - {dynamic_param_start_index}) / {dynamic_param_number}", 
            dynamic_length_name=dynamic_length_name,dynamic_param_start_index=dynamic_param_start_index, dynamic_param_number=dynamic_param_number);
        if !self._is_fuzzables_fixed_length() {
            res.push_str(
                format!(
                    "{indent}{every_dynamic_length};\n",
                    indent = indent,
                    every_dynamic_length = every_dynamic_length
                )
                .as_str(),
            );
        }

        let mut fixed_start_index = 0; //当前固定长度的变量开始分配的位置
        let mut dynamic_param_index = 0; //当前这是第几个动态长度的变量

        let fuzzable_param_number = self.fuzzable_params.len();
        for i in 0..fuzzable_param_number {
            let fuzzable_param = &self.fuzzable_params[i];
            let afl_helper = _AflHelpers::_new_from_fuzzable(fuzzable_param);
            let param_initial_line = afl_helper._generate_param_initial_statement(
                i,
                fixed_start_index,
                dynamic_param_start_index,
                dynamic_param_index,
                dynamic_param_number,
                &dynamic_length_name.to_string(),
                fuzzable_param,
            );
            res.push_str(
                format!(
                    "{indent}{param_initial_line}\n",
                    indent = indent,
                    param_initial_line = param_initial_line
                )
                .as_str(),
            );
            fixed_start_index = fixed_start_index + fuzzable_param._fixed_part_length();
            dynamic_param_index =
                dynamic_param_index + fuzzable_param._dynamic_length_param_number();
        }

        let mut test_function_call = format!(
            "{indent}test_function{test_index}(",
            indent = indent,
            test_index = test_index
        );
        for i in 0..fuzzable_param_number {
            if i != 0 {
                test_function_call.push_str(" ,");
            }
            test_function_call.push_str(format!("_param{}", i).as_str());
        }
        test_function_call.push_str(");\n");
        res.push_str(test_function_call.as_str());

        res
    }

    pub fn _to_well_written_function(
        &self,
        _api_graph: &ApiGraph,
        test_index: usize,
        indent_size: usize,
    ) -> String {
        let test_function_title = "fn test_function";
        let param_prefix = "_param";
        let local_param_prefix = "_local";
        let mut res = String::new();
        //生成对trait的引用
        let using_traits = self._generate_using_traits_string(indent_size);
        res.push_str(using_traits.as_str());
        //生成函数头
        let function_header = self._generate_function_header_string(
            _api_graph,
            test_index,
            indent_size,
            0,
            test_function_title,
            param_prefix,
        );
        res.push_str(function_header.as_str());

        //加入函数体开头的大括号
        res.push_str("{\n");

        //加入函数体
        if self._unsafe_tag {
            let unsafe_indent = _generate_indent(indent_size + 4);
            res.push_str(unsafe_indent.as_str());
            res.push_str("unsafe {\n");
            let unsafe_function_body = self._generate_function_body_string(
                _api_graph,
                indent_size + 4,
                param_prefix,
                local_param_prefix,
            );
            res.push_str(unsafe_function_body.as_str());
            res.push_str(unsafe_indent.as_str());
            res.push_str("}\n");
        } else {
            let function_body = self._generate_function_body_string(
                _api_graph,
                indent_size,
                param_prefix,
                local_param_prefix,
            );
            res.push_str(function_body.as_str());
        }

        //加入函数体结尾的大括号
        let braket_indent = _generate_indent(indent_size);
        res.push_str(braket_indent.as_str());
        res.push_str("}\n");

        res
    }

    pub fn _generate_using_traits_string(&self, indent_size: usize) -> String {
        let indent = _generate_indent(indent_size);
        let mut res = String::new();
        //using trait需要去重
        let mut has_used_traits = HashSet::new();
        for using_trait_ in &self._using_traits {
            if has_used_traits.contains(using_trait_) {
                continue;
            } else {
                has_used_traits.insert(using_trait_.clone());
            }
            res.push_str(indent.as_str());
            res.push_str("use ");
            res.push_str(using_trait_.as_str());
            res.push_str(";\n");
        }
        res.push('\n');
        res
    }

    //outer_indent:上层的缩进
    //extra_indent:本块需要的额外缩进
    pub fn _generate_function_header_string(
        &self,
        _api_graph: &ApiGraph,
        test_index: usize,
        outer_indent: usize,
        extra_indent: usize,
        test_function_title: &str,
        param_prefix: &str,
    ) -> String {
        let indent_size = outer_indent + extra_indent;
        let indent = _generate_indent(indent_size);

        //生成具体的函数签名
        let mut res = String::new();
        res.push_str(indent.as_str());
        res.push_str(test_function_title);
        res.push_str(test_index.to_string().as_str());
        res.push_str("(");

        //加入所有的fuzzable变量
        //第一个参数特殊处理
        let first_param = self.fuzzable_params.first();
        if let Some(first_param_) = first_param {
            if self._is_fuzzable_need_mut_tag(0) {
                res.push_str("mut ");
            }
            res.push_str(param_prefix);
            res.push('0');
            res.push_str(" :");
            res.push_str(first_param_._to_type_string().as_str());
        }

        let param_size = self.fuzzable_params.len();
        for i in 1..param_size {
            res.push_str(" ,");
            if self._is_fuzzable_need_mut_tag(i) {
                res.push_str("mut ");
            }
            let param = &self.fuzzable_params[i];
            res.push_str(param_prefix);
            res.push_str(i.to_string().as_str());
            res.push_str(" :");
            res.push_str(param._to_type_string().as_str());
        }
        res.push_str(") ");
        res
    }

    pub fn _generate_function_body_string(
        &self,
        _api_graph: &ApiGraph,
        outer_indent: usize,
        param_prefix: &str,
        local_param_prefix: &str,
    ) -> String {
        let extra_indent = 4;
        let mut res = String::new();
        let body_indent = _generate_indent(outer_indent + extra_indent);

        let dead_code = self._dead_code(_api_graph);

        //api_calls
        let api_calls_num = self.functions.len();
        let full_name_map = &_api_graph.full_name_map;
        let mut control_count = 0;
        let mut scope_flag = false;
        let mut unwrap_record_map: HashMap<String, String> = HashMap::new();
        let mut params_name: HashMap<usize, String> = HashMap::new();
        // let mut former_control_string = String::new();
        for i in 0..api_calls_num {
            let api_call = &self.functions[i];
            let (api_type, id) = api_call.func;
            //准备参数
            let param_size = api_call.params.len();
            let mut param_strings = Vec::new();
            // 处理Use的情况
            match api_type {
                ApiType::UseParam => {
                    for j in 0..param_size {
                        let (param_type, index, call_type) = &api_call.params[j];
                        if let Some(param_name) = params_name.get(index) {
                            // res.push_str(format!("println!(\"\{\}\", {})", ));   
                            let print_call = call_type._to_call_string(&param_name, full_name_map);
                            let print_line = format!("{}{};\n", body_indent, print_call);
                            res.push_str(print_line.as_str());
                        }
                    }
                    continue;
                }, 
                _ => {
                    
                }
            }
            for j in 0..param_size {
                let (param_type, index, call_type) = &api_call.params[j];
                let call_type_array = call_type._split_at_unwrap_call_type();
                //println!("call_type_array = {:?}",call_type_array);
                let param_name = match param_type {
                    ParamType::_FuzzableType => {
                        let mut s1 = param_prefix.to_string();
                        s1 += &(index.to_string());
                        s1
                    }
                    ParamType::_FunctionReturn => {
                        let mut s1 = local_param_prefix.to_string();
                        s1 += &(index.to_string());
                        s1
                    }
                };
                let call_type_array_len = call_type_array.len();
                if call_type_array_len == 1 {
                    let call_type = &call_type_array[0];
                    let param_string = call_type._to_call_string(&param_name, full_name_map);
                    param_strings.push(param_string);
                } else {
                    // 处理unwrap的情况，需要用到helper作为辅助
                    let mut former_param_name = param_name.clone();
                    let mut helper_index = 1;
                    let mut former_helper_line = String::new();
                    let mut unify_unwrap_flag = false;
                    for k in 0..call_type_array_len - 1 {
                        // let mut helper_indent = String::new();
                        let call_type = &call_type_array[k];
                        let unwrap_key = match call_type {
                            CallType::_UnwrapResult(inner_) => {
                                format!("{}{}", former_param_name, "unwrap_result")
                            }
                            CallType::_UnwrapOption(inner_) => {
                                format!("{}{}", former_param_name, "unwrap_option")
                            }
                            _ => {format!("Something Wrong")}
                        };
                        if unwrap_record_map.contains_key(&unwrap_key) {
                            // println!("unwrap record map contains: {}, {}", unwrap_key, k);
                            let helper_name = unwrap_record_map.get(&unwrap_key).unwrap();
                            former_helper_line = format!(
                                "{}let mut {} = {};\n",
                                body_indent,
                                helper_name,
                                call_type._to_call_string(&former_param_name, full_name_map)
                            );
                            former_param_name = helper_name.to_string();
                            helper_index = helper_index + 1;
                            if k == call_type_array_len - 2 {
                                unify_unwrap_flag = true;
                            }
                            continue;
                        }
                        let helper_name = format!(
                            "{}{}_param{}_helper{}",
                            local_param_prefix, i, j, helper_index
                        );
                        let helper_line = format!(
                            "{}let mut {} = {};\n",
                            body_indent,
                            helper_name,
                            call_type._to_call_string(&former_param_name, full_name_map)
                        );
                        if helper_index > 1 {
                            if !api_util::_need_mut_tag(call_type) {
                                former_helper_line = former_helper_line.replace("let mut ", "let ");
                            }
                            res.push_str(former_helper_line.as_str());
                        }
                        if helper_line.contains("_unwrap_result") {
                            let unwrap_key = format!("{}{}", former_param_name, "unwrap_result");
                            unwrap_record_map.insert(unwrap_key, helper_name.clone());
                        } else if helper_line.contains("_unwrap_option") {
                            let unwrap_key = format!("{}{}", former_param_name, "unwrap_option");
                            unwrap_record_map.insert(unwrap_key, helper_name.clone());
                        }
                        helper_index = helper_index + 1;
                        former_param_name = helper_name;
                        former_helper_line = helper_line;
                    }
                    let last_call_type = call_type_array.last().unwrap();
                    if !api_util::_need_mut_tag(last_call_type) {
                        former_helper_line = former_helper_line.replace("let mut ", "let ");
                    }
                    if !unify_unwrap_flag {
                        // println!("unify: false");
                        // if scope_flag { 
                        //     former_helper_line = former_helper_line.push_str(former_control_string.as_str());
                        //     res.replace(former_control_string.as_str(), former_helper_line.as_str());
                        // } else {
                        res.push_str(former_helper_line.as_str());
                        // }
                    }
                    let param_string =
                        last_call_type._to_call_string(&former_param_name, full_name_map);
                    param_strings.push(param_string);
                }
            }
            //如果不是最后一个调用
            res.push_str(body_indent.as_str());
            let (api_type, function_index) = &api_call.func;
            let api_function_index = api_call.func.1;
            let api_function = &_api_graph.api_functions[api_function_index];
            match api_type {
                ApiType::BareFunction | ApiType::GenericFunction => {
                    // 如果返回值是T，要类型声明？
                    // 这个T如果能推断出来，编译器也能自己推断出来吧？
                    if scope_flag {
                        res.push_str(body_indent.as_str());
                    }
                    if dead_code[i] || api_function._has_no_output() {
                        // res.push_str("let _ = ");
                    } else {
                        let mut_tag = if self._is_function_need_mut_tag(i) {
                            "mut "
                        } else {
                            ""
                        };
                        res.push_str(format!("let {}{}{} = ", mut_tag, local_param_prefix, i).as_str());
                        params_name.insert(i, format!("{}{}", local_param_prefix, i).to_string());
                    }
                    let api_function_full_name =
                        &_api_graph.api_functions[*function_index].full_name;
                    res.push_str(api_function_full_name.as_str());
                    res.push('(');
                    let param_size = param_strings.len();
                    for k in 0..param_size {
                        if k != 0 {
                            res.push_str(" ,");
                        }

                        let param_string = &param_strings[k];
                        res.push_str(param_string.as_str());
                    }
                    res.push_str(");\n");
                }
                ApiType::ControlStart => {
                    // res = res.replace("let _ = ", "");
                    scope_flag = true;
                    control_count += 1;
                    let mut control_string = String::new();
                    control_string.push_str("if ");
                    let param_size = param_strings.len();
                    for k in 0..param_size {
                        if k != 0 {
                            control_string.push_str(" && ");
                        }
                        let param_string = &param_strings[k];
                        // param_string.replace("", "");
                        control_string.push_str(param_string.as_str());
                    }
                    control_string.push_str(" {\n");
                    res.push_str(&control_string);
                    // former_control_string = control_string;
                }
                ApiType::ScopeEnd => {
                    // res = res.replace("let _ = ", "");
                    // res = res.replace(body_indent.as_str(), "");
                    scope_flag = false;
                    res.push_str("}\n");
                }
                ApiType::UseParam => {
                    
                }
            }
        }
        res
    }

    pub fn involved_unsafe_api(&self) -> Vec<usize>{
        let mut res = Vec::new();
        for i in 0..self.functions.len() {
            let function = &self.functions[i];
            if function.is_drop | function.is_get_raw_ptr | function.is_unsafe {
                // println!("Unsafe Function: {:#?}", function);
                res.push(function.func.1);
            }
        }
        res
    }


    pub fn delete_redundant_fuzzable_params(&mut self) {
        let mut fuzzable_params = self.fuzzable_params.clone();
        let mut record = vec![0; self.fuzzable_params.len()];
        // 第一次遍历，找到没有用过的fuzzable params
        for api_call in &self.functions {
            for param in &api_call.params {
                match param.0 {
                    ParamType::_FunctionReturn => {
                        
                    }
                    ParamType::_FuzzableType => {
                        record[param.1] = 1;
                    }
                }
            }
        }
        // 第二次倒序遍历，删除没有遍历过的fuzzable params
        let mut index = self.fuzzable_params.len();
        while index > 0 {
            index -= 1;
            if record[index] == 0 {
                fuzzable_params.remove(index);
            }
        }
        self.fuzzable_params = fuzzable_params;
        // 第三次遍历，更新api call的指向关系
        for api_call in &mut self.functions {
            for param in &mut api_call.params {
                match param.0 {
                    ParamType::_FunctionReturn => {
                        
                    },
                    ParamType::_FuzzableType => {
                        // let mut vec = record.clone();
                        // vec.split_off(param.1);
                        // let reduce = vec.into_iter().filter(|x| x == 0).collect().len();
                        let mut reduce = 0;
                        for i in 0..param.1 {
                            if record[i] == 0 {
                                reduce += 1;
                            }
                        }
                        param.1 -= reduce;
                    }
                }
            }
        }
    }

    // 判断该方法返回值是否被使用，如果被使用则返回最后调用的index
    pub fn func_is_called(&self, target_index: usize) -> Option<usize> {
        let mut call_index = 0;
        for i in 0..self.functions.len() {
            let api_call = &self.functions[i];
            for param in &api_call.params {
                match param.0 {
                    ParamType::_FunctionReturn => {
                        if param.1 == target_index {
                            call_index = i;
                        }
                    },
                    ParamType::_FuzzableType => {
                        continue;
                    }
                }
            }
        }
        if call_index == 0 {
            return None;
        } else {
            return Some(call_index);
        }
    }

    pub fn add_mut_flag(&mut self) {
        for api_call in &mut self.functions {
            for param in &mut api_call.params {
                match param.0 {
                    ParamType::_FunctionReturn => {
                        let call_type = &param.2;
                        let index = &param.1;
                        if api_util::_need_mut_tag(call_type) {
                            // println!("mut tag for {}", create_index);
                            self._function_mut_tag.insert(*index);
                        }
                    }
                    ParamType::_FuzzableType => {
                    }
                }
            }
        }
    }

    pub fn calc_output_type(&self) -> usize {
        let mut output_types = HashSet::new();
        for api_call in &self.functions {
            let output_type_ = &api_call.output_type;
            match output_type_ {
                Some(clean_) => {
                    output_types.insert(output_type_);
                },
                None => continue,
            }
        }
        output_types.len()
    }

    pub fn get_api_set(&self) -> HashSet<usize> {
        let mut api_set = HashSet::new();
        for api_call in &self.functions {
            api_set.insert(api_call.func.1);
        }
        api_set
    }

    pub fn get_api_list(&self) -> Vec<usize> {
        let mut api_list = Vec::new();
        for api_call in &self.functions {
            api_list.push(api_call.func.1);
        }
        api_list
    }
    
    pub fn get_last_api(&self) -> (ApiType, usize) {
        let seq_len = self.functions.len();
        self.functions[seq_len-1].func
    }

    pub fn is_sub_seq(&self, other_seq: &Self) -> bool {
        let api_list = self.get_api_list();
        let other_api_list = other_seq.get_api_list();
        let (mut i, mut j) = (0, 0);
        loop {
            if j == other_seq.len() {
                return true;
            }
            if i == self.len() {
                return false;
            }
            if api_list[i] == other_api_list[j] {
                i += 1;
                j += 1;
            } else {
                i += 1;
            }
        }
    }

    pub fn get_generic_path_number(&self, full_name: String) -> usize {
        let mut count = 0;
        for generic_path in self.generic_map.keys() {
            if full_name == generic_path.full_name {
                count += 1;
            }
        }
        count
    }

    pub fn get_generic_parameter(&self, generic_path_name: &String) -> Option<ApiParameter> {
        let generic_path = GenericPath::new(&generic_path_name, 0);
        if let Some(parameter) = self.generic_map.get(&generic_path) {
            return Some(parameter.clone());
        }
        None
    }

    pub fn add_generic_info(&mut self, generic_name: &String, parameter: ApiParameter) {
        // 以后再考虑同一个泛型可以对应多个Paramater
        self.generic_map.insert(GenericPath::new(generic_name, 0), parameter);
    }

    // 判断第i(return_index)行的返回值能否被第j(call_index)行所引用
    // 
    // ref_type:
    // 0, immut
    // 1, mut
    // pub fn return_usable_analyse_helper(
    //     &self,
    //     mut_map: &HashMap<usize, Vec<(usize, bool)>>,
    //     immut_map: &HashMap<usize, Vec<(usize, bool)>>,
    //     return_index: usize,
    //     call_index: usize,
    //     is_mut: bool,
    // ) -> bool
    // {
    //     if let Some(param_called_index) = self.func_is_called(return_index) {
    //         if param_called_index >= call_index {
    //             return false;
    //         }
    //     }
    //     match mut_map.get(&return_index) => {
    //         Some(mut_info) => {
    //             // 主要有mut引用，就可以分析了。
    //             // 被mut调用过，没被immut调用过
    //             // 1. 若flag为真，则继续借用分析
    //             // 2. 若flag为假，则usable
    //             for (param_index, flag) in &mut_info {
    //                 // 如果param_index小于call_index
    //                 // 则分析有没有mut immut同时借用的存在
    //                 if param_index < call_index && flag {
    //                     // 查看这个param_index存活了多久（决定了&mut 存活了多久）
    //                     // 如果比call_index要久，那么就是违法的
                    
    //                 }
    //                 if flag {
    //                     if !self.return_usable_analyse_helper(&mut_map, &immut_map, param_index, return_index, true) {
    //                         return false;
    //                     }
    //                 }
    //             }
    //             match immut_map.get(&i) => {
    //                 Some(immut_info) => {
    //                     // 
    //                 },
    //                 None => {
    //                     // mut引用没问题，且没有immut引用，则usable
    //                     return true;
    //                 }
    //             }
    //         },
    //         None => {
    //             match immut_map.get(&return_index) => {
    //                 Some(immut_info) => {
    //                     // 返回值调用过immut的return值
    //                     // 1. flag为真，判断该return值后续有没有被mut调用若没有，才是usable
    //                     // 2. flag为假，则usable
    //                     for (param_index, flag) in &immut_info {
    //                         if flag {
    //                             if !return_usable_analyse_helper(&mut_map, &immut_map, param_index, return_index, false) {
    //                                 return false;
    //                             }
    //                         }
    //                     }
    //                     return true;
    //                 },
    //                 None => {return true;} // 返回值没有调用过return值，usable
    //             }
    //         }
    //     }
    // }
}

impl fmt::Display for ApiSequence{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut function_sequence = String::new();
        let mut pattern_sequence = String::new();
        let l = self.functions.len();
        let mut arrow = 0;
        for i in 0 .. l {
            let api_call = &self.functions[i];
            match &api_call.func.0 {
                ApiType::BareFunction | ApiType::GenericFunction => {
                    if arrow != 0 {
                        function_sequence.push_str("->");
                        pattern_sequence.push_str("->");
                    }
                    arrow += 1;
                    function_sequence.push_str(&api_call.func.1.to_string());
                    let mut operate = String::from("[ ");
                    if api_call.is_unsafe{
                        operate.push_str("UnsafeConstruct ")
                    }
                    if api_call.is_get_raw_ptr{
                        operate.push_str("GetRawPtr ")
                    }
                    if api_call.is_drop{
                        operate.push_str("Drop ")
                    }
                    if api_call.is_mutate {
                        operate.push_str("Mutate ")
                    }
                    operate.push_str("]");
                    pattern_sequence = pattern_sequence + &operate;
                }
                ApiType::ControlStart => {
                    if arrow != 0 {
                        function_sequence.push_str("->");
                        pattern_sequence.push_str("->");
                    }
                    arrow = 0;
                    function_sequence.push_str("(");
                }
                ApiType::ScopeEnd => {
                    arrow += 1;
                    function_sequence.push_str(")");
                }
                ApiType::UseParam => {
                    function_sequence.push_str("(PRINTALL)");
                }
            }
        }
        write!(f, "{:?}, {:?}", function_sequence, pattern_sequence)
    }
}

use std::cmp::Ordering;
impl Ord for ApiSequence {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.len(), self.calc_output_type()).cmp(&(other.len(), other.calc_output_type()))
    }
}

impl PartialOrd for ApiSequence {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// impl PartialEq for ApiSequence {
//     fn eq(&self, other: &Self) -> bool {
//         (self.len(), self.calc_output_type()) == (other.len(), other.calc_output_type())
//     }
// }

// impl Eq for ApiSequence { }

pub fn _generate_indent(indent_size: usize) -> String {
    let mut indent = String::new();
    for _ in 0..indent_size {
        indent.push(' ');
    }
    indent
}