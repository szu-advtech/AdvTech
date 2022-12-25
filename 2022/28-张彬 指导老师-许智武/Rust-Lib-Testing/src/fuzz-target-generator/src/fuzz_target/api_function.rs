use crate::fuzz_target::api_util;
use crate::fuzz_target::call_type::CallType;
use crate::fuzz_target::fuzzable_type::{self, FuzzableType};
use crate::fuzz_target::impl_util::DefIdMap;
use crate::fuzz_target::operation_sequence::{StatementId, StatementInfo, StatememtSrc};
use crate::fuzz_target::impl_util::*;
use crate::fuzz_target::api_parameter;

use rustc_middle::mir::Safety;
use rustc_hir::{self, Mutability};
use rustc_span::Span;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::mir::Body;

use rustdoc::clean;
use rustdoc::clean::types::ItemId;

use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum ApiUnsafety {
    Unsafe,
    Normal,
}

#[derive(Clone, Debug)]
pub struct ApiFunction {
    pub full_name: String, //函数名，要来比较是否相等
    pub generics: HashMap<String, Vec<String>>, // 类型对应所需的特性 <T: Display + Debug>
    pub inputs: Vec<clean::Type>,
    pub output: Option<clean::Type>,
    pub _trait_full_path: Option<String>, //Trait的全限定路径,因为使用trait::fun来调用函数的时候，需要将trait的全路径引入
    pub _unsafe_tag: ApiUnsafety,
    pub def_id: ItemId,
    pub unsafe_info: HashMap<StatementId, StatementInfo>,
    pub rawptr_info: HashMap<StatementId, StatementInfo>,
    pub drop_info: HashMap<StatementId, StatementInfo>,
    pub mutate_info: HashMap<StatementId, StatementInfo>,
    pub return_info: HashMap<StatementId, StatementInfo>, // 用于判断返回值是否与输入相关（用于USE判断）
    pub func_types: HashSet<String>, // Func所包涉及到的所有数据类型
    pub input_types: Vec<String>,
    pub output_type: Option<String>,
    pub need_functions : HashMap<usize ,Vec<usize>>, // 所需要的第i个输入，可以通过这些function生成。前一个对应的index，后面对应的function下标。
    pub next_functions : HashSet<(usize, usize)>, // 生成的输出能够生成function的第i个输入。前一个对应的index，后面对应的function下标。
    pub weight: usize,
}

impl PartialEq for ApiFunction {
    fn eq(&self, other: &ApiFunction) -> bool {
        if self.full_name == other.full_name {
            return true;
        }
        false
    }
}

impl ApiUnsafety {
    pub fn _get_unsafety_from_fnheader(fn_header: &rustc_hir::FnHeader) -> Self {
        let unsafety = fn_header.unsafety;
        match unsafety {
            rustc_hir::Unsafety::Unsafe => ApiUnsafety::Unsafe,
            rustc_hir::Unsafety::Normal => ApiUnsafety::Normal,
        }
    }

    pub fn _is_unsafe(&self) -> bool {
        match self {
            ApiUnsafety::Unsafe => true,
            ApiUnsafety::Normal => false,
        }
    }
}

impl ApiFunction {

    /**
     * bohao
     * update need_functions for ApiFunction
     * need_input_index: input parameter index
     * fun: the function can offer the required input
     **/
    pub fn update_need_function(&mut self, need_input_index: usize, fun_index: usize) {
        let need_funs = self.need_functions.entry(need_input_index).or_insert(Vec::new());
        need_funs.push(fun_index);
    }

    /**
     * 
     */
    pub fn get_rawptr_struct_types(&self) -> Vec<String> {
        let mut struct_types = Vec::new();
        for (id, info) in &self.rawptr_info {
            match &info.src {
                Some(StatememtSrc::Init(init_context)) => struct_types.push(init_context.struct_type.clone()),
                Some(StatememtSrc::ParamSrc(param_src_context)) => struct_types.push(param_src_context.struct_type.clone()),
                Some(StatememtSrc::LocalSrc(_)) => {},
                Some(StatememtSrc::GlobalSrc(_)) => {},
                None => {},
            }
        }
        struct_types
    }

    pub fn get_drop_struct_types(&self) -> Vec<String> {
        let mut struct_types = Vec::new();
        for (id, info) in &self.drop_info {
            match &info.src {
                Some(StatememtSrc::Init(init_context)) => struct_types.push(init_context.struct_type.clone()),
                Some(StatememtSrc::ParamSrc(param_src_context)) => struct_types.push(param_src_context.struct_type.clone()),
                Some(StatememtSrc::LocalSrc(_)) => {},
                Some(StatememtSrc::GlobalSrc(_)) => {},
                None => {},
            }
        }
        struct_types
    }

    pub fn get_all_struct_types(&self) -> Vec<String> {
        let mut struct_types = Vec::new();
        for type_ in &self.func_types {
            struct_types.push(type_.clone());
        }
        struct_types
    }

    /**
     * bohao
     * update next_functions for ApiFunction
     * next_input_index: offer input parameter index for the next function
     * fun: the function can accept the output of the function as input
     **/
    pub fn update_next_function(&mut self, next_input_index: usize, fun_index: usize) {
        self.next_functions.insert((next_input_index, fun_index));
    }

    pub fn is_unsafe_function(&self) -> bool{
        if !self.unsafe_info.is_empty() | self._unsafe_tag._is_unsafe() {
            return true;
        }
        false
    }

    pub fn _is_end_function(&self, full_name_map: &FullNameMap) -> bool {
        if self.contains_mut_borrow() {
            return false;
        }
        let return_type = &self.output;
        match return_type {
            Some(ty) => {
                if api_util::_is_end_type(&ty, full_name_map) {
                    return true;
                } else {
                    return false;
                }
            }
            None => true,
        }
        //TODO:考虑可变引用或者是可变裸指针做参数的情况
    }

    pub fn contains_mut_borrow(&self) -> bool {
        let input_len = self.inputs.len();
        if input_len <= 0 {
            return false;
        }
        for input_type in &self.inputs {
            match input_type {
                clean::Type::BorrowedRef { mutability, .. }
                | clean::Type::RawPointer(mutability, _) => {
                    if let Mutability::Mut = mutability {
                        return true;
                    }
                }
                _ => {}
            }
        }
        return false;
    }

    pub fn _is_start_function(&self, full_name_map: &FullNameMap) -> bool {
        let input_types = &self.inputs;
        let mut flag = true;
        for ty in input_types {
            if !api_util::_is_end_type(&ty, full_name_map) {
                flag = false;
                break;
            }
        }
        flag
    }

    //TODO:判断一个函数是否是泛型函数
    pub fn _is_generic_function(&self) -> bool {
        let input_types = &self.inputs;
        for ty in input_types {
            if api_parameter::is_generic_type(&ty) {
                return true;
            }
        }
        let output_type = &self.output;
        if let Some(ty) = output_type {
            if api_parameter::is_generic_type(&ty) {
                return true;
            }
        }
        return false;
    }

    pub fn _is_generic_output(&self) -> bool {
        let output_type = &self.output;
        if let Some(ty) = output_type {
            if api_parameter::is_generic_type(&ty) {
                return true;
            }
        }
        return false;
    }

    pub fn _is_generic_input(&self, param_index: usize) -> bool {
        let input_type = &self.inputs[param_index];
        api_parameter::is_generic_type(input_type)
    }

    pub fn _has_no_output(&self) -> bool {
        match self.output {
            None => true,
            Some(_) => false,
        }
    }

    pub fn _pretty_print(&self, full_name_map: &FullNameMap) -> String {
        let mut fn_line = format!("fn {}(", self.full_name);
        let input_len = self.inputs.len();
        for i in 0..input_len {
            let input_type = &self.inputs[i];
            if i != 0 {
                fn_line.push_str(" ,");
            }
            fn_line.push_str(api_util::_type_name(input_type, full_name_map).as_str());
        }
        fn_line.push_str(")");
        if let Some(ref ty_) = self.output {
            fn_line.push_str("->");
            fn_line.push_str(api_util::_type_name(ty_, full_name_map).as_str());
        }
        fn_line
    }

    pub fn filter_by_fuzzable_type(&self, full_name_map: &FullNameMap) -> bool {
        for input_ty_ in &self.inputs {
            if api_util::is_fuzzable_type(input_ty_, full_name_map) {
                let fuzzable_call_type =
                    fuzzable_type::fuzzable_call_type_by_clean_type(input_ty_, full_name_map);
                let (fuzzable_type, call_type) =
                    fuzzable_call_type.generate_fuzzable_type_and_call_type();

                match &fuzzable_type {
                    FuzzableType::NoFuzzable => {
                        return true;
                    }
                    _ => {}
                }

                if fuzzable_type._is_multiple_dynamic_length() {
                    return true;
                }

                match &call_type {
                    CallType::_NotCompatible => {
                        return true;
                    }
                    _ => {}
                }
            }
        }
        return false;
    }

    pub fn set_info(&mut self, unsafe_info: HashMap<StatementId, StatementInfo>, rawptr_info: HashMap<StatementId, StatementInfo>, drop_info: HashMap<StatementId, StatementInfo>, func_types: HashSet<String>){
        self.unsafe_info = unsafe_info;
        self.rawptr_info = rawptr_info;
        self.drop_info = drop_info;
        self.func_types = func_types;
    }

    // pub fn add_input_type_name(&mut self, body: Body<'_>) {
    //     let ItemId::DefId(def_id_) = self.def_id;
    //     let local_def_id = LocalDefId {
    //         local_def_index: def_id_.index,
    //     };
    //     self.input_types.push("".to_string())
    // }
    
    // 是否需要考虑函数返回值被drop掉的情况
    pub fn is_nonprimitive_return(&self) -> bool {
        if let Some(output_type) = &self.output {
            return !output_type.is_primitive();
        }
        false
    }

    // 是否和返回值相关
    pub fn return_relate_param(&self) -> Option<usize> {
        if let Some(output_ty) = &self.output {
            if output_ty.is_primitive() {
                return None;
            }
            let inner_ty = get_inner_type(output_ty); 
            match inner_ty {
                clean::Type::BorrowedRef {..} | clean::Type::RawPointer(..) | clean::Type::DynTrait(..) => { },
                _ => {return None;},
            }
            let return_info = &self.return_info;
            // println!("{}\ninner ty: {:#?}\nreturn info: {:#?}", self.full_name, inner_ty, return_info);
            for (id, info) in return_info {
                match &info.src {
                    Some(StatememtSrc::Init(init_context)) => {
                        let local_num = init_context.local.as_usize();
                        if local_num <= self.inputs.len() {
                            return Some(local_num);
                        }
                    }
                    Some(StatememtSrc::ParamSrc(param_context)) => {
                        let local_num = param_context.local.as_usize();
                        if local_num <= self.inputs.len() {
                            return Some(local_num);
                        }
                    }
                    Some(StatememtSrc::LocalSrc(local_context)) => {
                        let local_num = local_context.local.as_usize();
                        if local_num <= self.inputs.len() {
                            return Some(local_num);
                        }
                    }
                    Some(StatememtSrc::GlobalSrc(global_context)) => {
                        let local_num = global_context.local.as_usize();
                        if local_num <= self.inputs.len() {
                            return Some(local_num);
                        }
                    }
                    _ => {}
                }
            }
        }
        None
    }
    
    // 通过symbol找到trait_bound
    pub fn get_trait_bound_by_symbol(&self, symbol: &String) -> Option<Vec<String>>{
        // println!("function: {}", self.full_name);
        for (generic_path, trait_bound) in &self.generics {
            let paths: Vec<&str> = generic_path.as_str().split("::").collect();
            // println!("{:?}\n{:?}", generic_path, paths);
            if symbol == paths[paths.len()-1] {
                return Some(trait_bound.to_vec());
            }
        }
        None
    }

    // 通过clean::type获得generic_path:
    pub fn get_generic_path_by_symbol(&self, symbol: &String) -> Option<String> {
        // println!("function: {}", self.full_name);
        for (generic_path, trait_bound) in &self.generics {
            let paths: Vec<&str> = generic_path.as_str().split("::").collect();
            // println!("{:?}\n{:?}", generic_path, paths);
            if symbol == paths[paths.len()-1] {
                return Some(generic_path.to_string());
            }
        }
        None
    }
    
    // 获得generic_path_name
    pub fn get_generic_path_by_param_index(&self, param_index: usize) -> Option<String> {
        if param_index >= self.inputs.len() {
            return None;
        }
        let type_ = &self.inputs[param_index];
        if let Some(symbol) = api_parameter::get_generic_name(type_) {
            return self.get_generic_path_by_symbol(&symbol);
        }
        return None;
    }

    // 获得generic_path_name
    pub fn get_generic_path_for_output(&self) -> Option<String> {
        if let Some(type_) = &self.output {
            if let Some(symbol) = api_parameter::get_generic_name(type_) {
                return self.get_generic_path_by_symbol(&symbol);
            }
            return None;
        }
        return None;
    }

    pub fn format(&self) -> String {
        let function_name = &self.full_name;
        let mut input_str = String::from("(");
        let input_len = self.input_types.len();
        for i in 0..input_len {
            let type_ = &self.input_types[i];
            input_str += type_;
            if i != input_len - 1 {
                input_str += ", "
            }
        }
        input_str += ")";

        let mut output_str = String::from("");
        match &self.output_type {
            Some(type_) => {
                output_str += " -> ";
                output_str += type_;
            },
            None => {}
        }

        let mut tag_str = String::from("{ ");
        if self.unsafe_info.keys().len() > 0 {
            tag_str += "UNSAFE"
        }
        if self.rawptr_info.keys().len() > 0 {
            tag_str += "GETRAWPTR "
        }
        if self.drop_info.keys().len() > 0 {
            tag_str += "DROP "
        }
        if self.mutate_info.keys().len() > 0 {
            tag_str += "MUTATE "
        }
        tag_str += "}";

        format!("{} {}{} {}", function_name, input_str, output_str, tag_str)
    }
}

use std::fmt;
impl fmt::Display for ApiFunction{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        


        write!(f, "{}", &self.format())
        // fn (input1, input2) -> output { MUTATE ... }
    }
}