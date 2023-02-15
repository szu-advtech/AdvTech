use rustdoc::clean::{self, ItemKind, types::{GenericArg, GenericArgs, Type}};
use rustc_hir::{self, def, def_id::DefId};
use rustc_span::Symbol;
use rustc_data_structures::thin_vec::ThinVec;
use crate::fuzz_target::fuzzable_type::{self, FuzzableType};
use crate::fuzz_target::impl_util::FullNameMap;
use crate::fuzz_target::api_util;

use std::collections::{HashMap, HashSet};
use std::boxed::Box;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ApiParameter {
    Struct(ApiStructure),
    Enum(ApiStructure),
    Preluded(PreludedStructure),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct GenericPath {
    pub full_name: String,
    pub number: usize,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ApiStructure {
    pub name: String, // 结构名，包含层级关系
    pub is_enum: bool,
    pub implemented_traits: Vec<String>, // Display, Debug, Clone...
    pub is_generic: bool, // 是否为泛型
    // pub is_generable: bool, // 是否可生成
    pub return_functions: Vec<usize>, // 未考虑Tuple
    pub use_functions: Vec<(usize, usize)>, // (funciton_index, param_index)
    pub clean_ty: Option<clean::Type>,
    pub implemented_functions: Vec<String>, // 包括结构体实现的方法和trait方法
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PreludedStructure {
    pub name: String,
    pub preluded_type: FuzzableType,
    pub implemented_traits: Vec<String>,
    pub use_functions: Vec<(usize, usize)>,
    pub implemented_functions: Vec<String>,
}

impl PreludedStructure {
    pub fn new(fuzzable_type: &FuzzableType) -> Option<Self> {
        let preluded_type = fuzzable_type.clone();
        let name = fuzzable_type.as_string_name();
        let mut implemented_traits = Vec::new();
        let use_functions = Vec::new();
        let implemented_functions = Vec::new();
        match fuzzable_type {
            FuzzableType::NoFuzzable | FuzzableType::Use => { return None; }
            FuzzableType::String => {
                // String实现的traits，在目标crate实现的将后续加入
                implemented_traits.push(String::from("core::fmt::Display"));
                implemented_traits.push(String::from("core::fmt::Debug"));
                implemented_traits.push(String::from("core::convert::From"));
                implemented_traits.push(String::from("core::clone::Clone"));
                implemented_traits.push(String::from("core::copy::Copy"));
                // use_funcitons 需要在后续更新
                // implemented_fuctions 需要在后续更新
                Some(
                    PreludedStructure {
                        name,
                        preluded_type,
                        implemented_traits,
                        use_functions,
                        implemented_functions,
                    }
                )
            }
            _ => {
                Some(
                    PreludedStructure {
                        name,
                        preluded_type,
                        implemented_traits,
                        use_functions,
                        implemented_functions,
                    }
                )
            }
        }
    }

    pub fn is_same_fuzzable_type(&self, fuzzable_type: &FuzzableType) -> bool {
        if &self.preluded_type == fuzzable_type {
            return true;
        }
        false
    }
}

impl ApiParameter {
    // TODO: 考虑预设数据类型
    pub fn get_parameter_traits(&self) -> Vec<String> {
        match &self {
            ApiParameter::Struct(api_struct)
            | ApiParameter::Enum(api_struct) => api_struct.implemented_traits.clone(),
            ApiParameter::Preluded(preluded_struct) => preluded_struct.implemented_traits.clone(),
        }
    }

    pub fn as_string(&self) -> String {
        match &self {
            ApiParameter::Struct(api_struct) => format!("Struct({})", api_struct.name.clone()),
            ApiParameter::Enum(api_struct) => format!("Enum({})", api_struct.name.clone()),
            ApiParameter::Preluded(preluded_struct) => format!("Preluded({})", preluded_struct.name.clone()),
        }
    }

    pub fn from_api_struct(api_struct: ApiStructure) -> Self {
        match api_struct.is_enum {
            true => ApiParameter::Enum(api_struct),
            false => ApiParameter::Struct(api_struct),
        }
    }

    pub fn from_preluded_struct(fuzzable_struct: PreludedStructure) -> Self {
        ApiParameter::Preluded(fuzzable_struct)
    }

    pub fn is_meet_traits(&self, traits: &Vec<String>) -> bool {
        let implemented_traits = match self {
            ApiParameter::Struct(api_struct) | ApiParameter::Enum(api_struct) => {
                &api_struct.implemented_traits
            },
            ApiParameter::Preluded(preluded_struct) => {
                &preluded_struct.implemented_traits
            },
        };
        let minusion = traits.iter().filter(|&u| !implemented_traits.contains(u)).collect::<Vec<_>>();
        println!("minusion: {:?}", minusion);
        if minusion.len() > 0 {
            return false;
        }
        true
    }

    pub fn is_same_type(&self, clean_ty: &clean::Type, full_name_map: &FullNameMap) -> bool {
        match self {
            ApiParameter::Struct(api_struct) | ApiParameter::Enum(api_struct) => {
                return api_struct.is_same_clean_type(clean_ty);
            },
            // Preluded Struct 不用保留clean type
            ApiParameter::Preluded(preluded_struct) => {
                // 通过判断clean_ty是否为fuzzable再进一步判断
                let fuzzable_call_type = fuzzable_type::fuzzable_call_type_by_clean_type(clean_ty, full_name_map);
                let (fuzzable_type, _) = fuzzable_call_type.generate_fuzzable_type_and_call_type();
                return preluded_struct.is_same_fuzzable_type(&fuzzable_type);
            },
        }
    }

    pub fn add_use_function(&mut self, func_index: usize, param_index: usize) {
        let func_param_tuple = (func_index, param_index);
        match self {
            ApiParameter::Struct(api_struct) | ApiParameter::Enum(api_struct) => {
                if !api_struct.use_functions.contains(&func_param_tuple) {
                    api_struct.use_functions.push(func_param_tuple);
                }
            },
            ApiParameter::Preluded(preluded_struct) => {
                if !preluded_struct.use_functions.contains(&func_param_tuple) {
                    preluded_struct.use_functions.push(func_param_tuple);
                }
            },
        }
    }

    pub fn add_return_function(&mut self, index: usize) {
        match self {
            ApiParameter::Struct(api_struct) | ApiParameter::Enum(api_struct) => {
                if !api_struct.return_functions.contains(&index) {
                    api_struct.return_functions.push(index);
                }
            },
            // Preluded通过自定义变量生成，不需要考虑返回函数信息
            ApiParameter::Preluded(..) => {},
        }
    }
    
    pub fn get_use_functions(&self) -> Vec<(usize, usize)> {
        match self {
            ApiParameter::Struct(api_struct) | ApiParameter::Enum(api_struct) => {
                return api_struct.use_functions.clone();
            },
            ApiParameter::Preluded(preluded_struct) => { 
                return preluded_struct.use_functions.clone(); 
            },
        }
    }

    pub fn get_return_functions(&self) -> Vec<usize> {
        match self {
            ApiParameter::Struct(api_struct) | ApiParameter::Enum(api_struct) => {
                return api_struct.return_functions.clone();
            },
            ApiParameter::Preluded(..) => { return Vec::new(); },
        }
    }

    pub fn is_preluded(&self) -> bool {
        match self {
            ApiParameter::Preluded(..) => true,
            _ => false,
        }
    }

    pub fn is_struct(&self) -> bool {
        match self {
            ApiParameter::Struct(..) => true,
            _ => false,
        }
    }

    pub fn is_enum(&self) -> bool {
        match self {
            ApiParameter::Enum(..) => true,
            _ => false,
        }
    }

    pub fn is_generic(&self) -> bool {
        match self {
            ApiParameter::Struct(api_struct) | ApiParameter::Enum(api_struct) => {
                return api_struct.is_generic;
            },
            ApiParameter::Preluded(..) => {
                return false;
            }
        }
    }

    // 结果是否为泛型（包括子元素）
    pub fn judge_generic(&mut self) {
        match self {
            ApiParameter::Struct(api_struct) | ApiParameter::Enum(api_struct) => {
                if let Some(clean_type) = &api_struct.clean_ty {
                    if api_util::_is_generic_type(&clean_type) {
                        api_struct.is_generic = true;
                    }
                } else {
                    panic!("can't judge structure generic");
                }
            },
            ApiParameter::Preluded(..) => { }
        }
    }

    pub fn set_clean_type(&mut self, ty_: &clean::Type) {
        match self {
            ApiParameter::Struct(api_struct) | ApiParameter::Enum(api_struct) => {
                api_struct.set_clean_type(ty_);
            },
            ApiParameter::Preluded(..) => { },
        }
    }

    pub fn judge_enum(&mut self) {
        match self {
            ApiParameter::Struct(api_struct) => {
                api_struct.judge_enum();
                if api_struct.is_enum {
                    *self = ApiParameter::Enum(api_struct.clone())
                }
            },
            ApiParameter::Enum(..) | ApiParameter::Preluded(..) => { },
        }
    }

    pub fn add_implemented_trait(&mut self, trait_: &String) {
        match self {
            ApiParameter::Struct(param) 
            | ApiParameter::Enum(param) => {
                if !param.implemented_traits.contains(trait_) {
                    param.implemented_traits.push(trait_.clone());
                }
            }
            ApiParameter::Preluded(preluded_param) => {
                if !preluded_param.implemented_traits.contains(trait_) {
                    preluded_param.implemented_traits.push(trait_.clone());
                }
            }
        }
    }

    pub fn add_implemented_function(&mut self, func_: &String) {
        match self {
            ApiParameter::Struct(param) 
            | ApiParameter::Enum(param) => {
                if !param.implemented_functions.contains(func_) {
                    param.implemented_functions.push(func_.clone());
                }
            }
            ApiParameter::Preluded(preluded_param) => {
                if !preluded_param.implemented_functions.contains(func_) {
                    preluded_param.implemented_functions.push(func_.clone());
                }
            }
        }
    }

    pub fn is_implement_debug_trait(&self) -> bool {
        match self {
            ApiParameter::Struct(param) 
            | ApiParameter::Enum(param) => {
                param.implemented_traits.contains(&String::from("core::fmt::Debug"))
            }
            ApiParameter::Preluded(preluded_param) => {
                preluded_param.implemented_traits.contains(&String::from("core::fmt::Debug"))
            }
        }
    }

    pub fn is_implement_display_trait(&self) -> bool {
        match self {
            ApiParameter::Struct(param) 
            | ApiParameter::Enum(param) => {
                param.implemented_traits.contains(&String::from("core::fmt::Display"))
            }
            ApiParameter::Preluded(preluded_param) => {
                preluded_param.implemented_traits.contains(&String::from("core::fmt::Display"))
            }
        }
    }

    pub fn is_fuzzable(&self) -> bool {
        match self {
            ApiParameter::Struct(param) 
            | ApiParameter::Enum(param) => {
                false
            }
            ApiParameter::Preluded(preluded_param) => {
                true
            }
        }
    }
}

impl GenericPath {
    pub fn new(name: &String, count: usize) -> GenericPath {
        GenericPath {
            full_name: name.clone(),
            number: count,
        }
    }

    pub fn as_string(&self) -> String {
        format!("{}-{}", self.full_name, self.number)
    }
}

impl ApiStructure {
    // 通过结构名和是否为泛型结构新建
    pub fn new(struct_name: String) -> ApiStructure {
        ApiStructure {
            name: struct_name,
            is_enum: false,
            implemented_traits: Vec::new(),
            is_generic: false,
            return_functions: Vec::new(),
            use_functions: Vec::new(),
            clean_ty: None,
            implemented_functions: Vec::new(),
        }
    }

    pub fn new_primitive() -> ApiStructure {
        ApiStructure {
            name: "".to_string(),
            is_enum: false,
            implemented_traits: Vec::new(),
            is_generic: false,
            return_functions: Vec::new(),
            use_functions: Vec::new(),
            clean_ty: None,
            implemented_functions: Vec::new(),
        }
    }

    // 设置clean::Type信息
    // 如果已经设置则不重复设置
    pub fn set_clean_type(&mut self, ty_: &clean::Type) {
        if self.clean_ty == None {
            self.clean_ty = Some(ty_.clone());
        } else {
            println!("Origin Type: {:#?}", self.clean_ty);
            println!("New Type: {:#?}", ty_);
            // assert!(self.clean_ty == Some(ty_.clone()));
        }
    }

    // 获取clean::Type信息
    // pub fn get_clean_type(&self) -> Option<clean::Type> {
    //     self.clean_ty.as_ref()
    // }

    // 判断该结构体是否可以生成
    pub fn is_generable(&self) -> bool {
        self.is_generic
    }

    // 判断泛型方法能否使用该结构体

    // 判断是否和clean相同
    pub fn is_same_clean_type(&self, clean_type: &clean::Type) -> bool {
        let clean_type = &get_inner_type(clean_type);
        if let Some(clean_ty) = &self.clean_ty {
            // println!("{:#?}", clean_ty);
            match (clean_ty, clean_type) {
                // 这种情况只比较res
                (clean::Type::Path{path: path1}, clean::Type::Path{path: path2}) => {
                    return path1.res == path2.res;
                }
                (a, b) => {
                    return a == b;
                }
            }
        }
        false
    }

    // 如果是enum类型则设置is_enum为真
    // 通过clean_ty来判断
    pub fn judge_enum(&mut self) {
        if let Some(clean_type) = &self.clean_ty {
            match clean_type {
                clean::Type::Path { path, .. } => {
                    match path.res {
                        def::Res::Def(def::DefKind::Enum, ..) => self.is_enum = true,
                        _ => {}
                    }
                },
                _ => {}
            }
        }
    }
}

// 只看表层。
pub fn is_generic_type(ty: &clean::Type) -> bool {
    //TODO：self不需要考虑，因为在产生api function的时候就已经完成转换，但需要考虑类型嵌套的情况
    match ty {
        clean::Type::Generic(_) => true,
        clean::Type::Path{path} => {
            if path.segments[0].name.as_str() == "Option" || path.segments[0].name.as_str() == "Result" {
                match &path.segments[0].args {
                    GenericArgs::AngleBracketed{args, bindings} => {
                        match &args[0] {
                            GenericArg::Type(inner_type) => {
                                return is_generic_type(&inner_type);
                            },
                            _ => false,
                        }
                    },
                    GenericArgs::Parenthesized{inputs, output} => {
                        return is_generic_type(&inputs[0]);
                    }
                }
            } else {
                return false;
            }
        },
        clean::Type::Tuple(types) => {
            // 目前只考虑tuple的第一个
            if types.len() > 0 {
                return is_generic_type(&types[0]);
            }
            return false;
        }
        clean::Type::Slice(type_)
        | clean::Type::Array(type_, ..)
        | clean::Type::RawPointer(_, type_)
        | clean::Type::BorrowedRef { type_, .. } => {
            let inner_type = &**type_;
            return is_generic_type(inner_type);
        }
        _ => {
            return false;
        }
    }
}

// 只看表层，不看path内部(除了Option, Result)
pub fn get_inner_type(type_: &clean::Type) -> clean::Type {
    match type_ {
        clean::Type::Tuple(vec_type) => {
            if vec_type.len() > 0 {
                return get_inner_type(&vec_type[0]);
            } else {
                println!("Tuple strange type: {:#?}", type_);
                return type_.clone();
            }
        }
        clean::Type::Slice(box_type) => get_inner_type(&*box_type),
        clean::Type::Array(box_type, s) => get_inner_type(&*box_type),
        clean::Type::RawPointer(_, type_)
        | clean::Type::BorrowedRef { type_, .. } => get_inner_type(&*type_),
        clean::Type::Path{path} => {
            if path.segments[0].name.as_str() == "Option" || path.segments[0].name.as_str() == "Result" {
                match &path.segments[0].args {
                    GenericArgs::AngleBracketed{args, bindings} => {
                        match &args[0] {
                            GenericArg::Type(inner_type) => {
                                get_inner_type(&inner_type)
                            },
                            _ => type_.clone()
                        }
                    },
                    GenericArgs::Parenthesized{inputs, output} => {
                        get_inner_type(&inputs[0])
                    }
                }
            } else {
                type_.clone()
            }
        },
        _ => type_.clone(),
    }
}

pub fn replace_generic_type(origin_type: &clean::Type, new_type: &clean::Type) -> clean::Type {
    match origin_type {
        clean::Type::BorrowedRef { lifetime, mutability, type_ }
        => {   
            let inner_type = &**type_;
            if is_generic_type(inner_type) {
                let replaced_type = replace_generic_type(inner_type, new_type);
                return clean::Type::BorrowedRef {
                    lifetime: lifetime.clone(),
                    mutability: mutability.clone(),
                    type_: Box::new(replaced_type),
                };
            } else {
                return origin_type.clone();
            }
        }
        clean::Type::Generic(..) => {
            return new_type.clone();
        }
        clean::Type::Path{path} => {
            if path.segments[0].name.as_str() == "Option" || path.segments[0].name.as_str() == "Result" {
                let clean::Path { res, segments } = path;
                //let clean::Path {global, res, segments} = path;
                let mut new_segments = Vec::new();
                for path_segment in segments {
                    let clean::PathSegment {
                        name,
                        args: generic_args,
                    } = path_segment;
                    match generic_args {
                        clean::GenericArgs::AngleBracketed { args, bindings } => {
                            let mut new_args = Vec::new();
                            for generic_arg in args {
                                if let clean::GenericArg::Type(generic_type) = generic_arg {
                                    if is_generic_type(generic_type) {
                                        let replaced_type = replace_generic_type(generic_type, new_type);
                                        let new_generic_arg = clean::GenericArg::Type(replaced_type);
                                        new_args.push(new_generic_arg);
                                    } else {
                                        new_args.push(generic_arg.clone());
                                    }
                                } else {
                                    new_args.push(generic_arg.clone());
                                }
                            }
                            let new_generic_args = clean::GenericArgs::AngleBracketed {
                                args: new_args,
                                bindings: bindings.clone(),
                            };
                            let new_path_segment = clean::PathSegment {
                                name: name.clone(),
                                args: new_generic_args,
                            };
                            new_segments.push(new_path_segment);
                        }
                        clean::GenericArgs::Parenthesized { inputs, output } => {
                            let mut new_inputs = Vec::new();
                            for input_type in inputs {
                                if is_generic_type(input_type) {
                                    let replaced_type = replace_generic_type(input_type, new_type);
                                    new_inputs.push(replaced_type);
                                } else {
                                    new_inputs.push(input_type.clone());
                                }
                            }
                            let new_output = match output {
                                None => None,
                                Some(output_type) => {
                                    let new_output_type = if is_generic_type(output_type) {
                                        let replaced_type =
                                            Box::new(replace_generic_type(output_type, new_type));
                                        replaced_type
                                    } else {
                                        output_type.clone()
                                    };
                                    Some(new_output_type)
                                }
                            };
                            let new_generic_args = clean::GenericArgs::Parenthesized {
                                inputs: new_inputs,
                                output: new_output,
                            };
                            let new_path_segment = clean::PathSegment {
                                name: name.clone(),
                                args: new_generic_args,
                            };
                            new_segments.push(new_path_segment);
                        }
                    }
                }
                let new_path = clean::Path {
                    //global: global.clone(),
                    res: res.clone(),
                    segments: new_segments,
                };
                let new_type = clean::Type::Path {
                    path: new_path,
                    //is_generic: is_generic.clone(),
                };
                return new_type.clone();
            } else {
                return origin_type.clone();
            }
        },
        clean::Type::Tuple(types) => {
            let mut new_tuple_type = types.clone();
            for i in 0..types.len() {
                let replaced_type = replace_generic_type(&types[i], new_type);
                new_tuple_type[i] = replaced_type;
            }
            return clean::Type::Tuple(new_tuple_type);
        }
        clean::Type::RawPointer(mutability, type_) => {
            let inner_type = &**type_;
            let replaced_type = replace_generic_type(inner_type, new_type);
            return clean::Type::RawPointer(mutability.clone(), Box::new(replaced_type));
        }
        clean::Type::Array(type_, string) => {
            let inner_type = &**type_;
            let replaced_type = replace_generic_type(inner_type, new_type);
            return clean::Type::Array(Box::new(replaced_type), string.clone());
        }
        clean::Type::Slice(type_) => {
            let inner_type = &**type_;
            let replaced_type = replace_generic_type(inner_type, new_type);
            return clean::Type::Slice(Box::new(replaced_type));
        }
        _ => {
            return origin_type.clone();
        }
    }
}

pub fn get_generic_name(ty: &clean::Type) -> Option<String> {
    //TODO：self不需要考虑，因为在产生api function的时候就已经完成转换，但需要考虑类型嵌套的情况
    match ty {
        clean::Type::Generic(symbol) => {
            return Some(symbol.as_str().to_string());
        },
        clean::Type::Path{path} => {
            if path.segments[0].name.as_str() == "Option" || path.segments[0].name.as_str() == "Result" {
                match &path.segments[0].args {
                    GenericArgs::AngleBracketed{args, bindings} => {
                        match &args[0] {
                            GenericArg::Type(inner_type) => {
                                return get_generic_name(&inner_type);
                            },
                            _ => {return None;},
                        }
                    },
                    GenericArgs::Parenthesized{inputs, output} => {
                        return get_generic_name(&inputs[0]);
                    }
                }
            } else {
                return None;
            }
        },
        clean::Type::Tuple(types) => {
            for ty_ in types {
                if is_generic_type(ty_) {
                    return get_generic_name(ty_);
                }
            }
            return None;
        }
        clean::Type::Slice(type_)
        | clean::Type::Array(type_, ..)
        | clean::Type::RawPointer(_, type_)
        | clean::Type::BorrowedRef { type_, .. } => {
            let inner_type = &**type_;
            return get_generic_name(inner_type);
        }
        _ => {
            return None;
        }
    }
}