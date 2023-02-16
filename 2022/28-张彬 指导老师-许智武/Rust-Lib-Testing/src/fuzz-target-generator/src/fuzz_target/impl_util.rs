use crate::fuzz_target::api_util;
use rustc_hir::def_id::DefId;
use rustdoc::clean::{self, ItemKind, types::{GenericArg, GenericArgs, Type}};
use rustdoc::formats::cache::Cache;
use std::collections::{HashMap,HashSet};
// use rustdoc::html::item_type::ItemType;
use crate::fuzz_target::api_function::ApiFunction;
use crate::fuzz_target::collector::*;
use rustdoc::formats::item_type::ItemType;
use rustdoc::core::DocContext;

use rustc_hir::def_id::{LOCAL_CRATE, LocalDefId};
use rustc_hir::FnDecl;
use rustc_hir::{self, Node};
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;
use rustc_middle::mir::{StatementKind, Body, Local, LocalInfo, Place, ProjectionElem, Rvalue, ClearCrossCrate, Safety};
//TODO:是否需要为impl里面的method重新设计数据结构？目前沿用了ApiFunction,或者直接对ApiFunction进行扩展
//两种函数目前相差一个defaultness
use crate::fuzz_target::api_function::ApiUnsafety;
use crate::fuzz_target::api_graph::ApiGraph;
use crate::fuzz_target::api_parameter::{ApiStructure, ApiParameter};
use crate::fuzz_target::prelude_type;
use std::boxed::Box;

#[derive(Debug, Clone)]
pub struct CrateImplCollection {
    //impl type类型的impl块
    pub impl_types: Vec<clean::Impl>,
    //impl type for trait类型的impl块
    pub impl_trait_for_types: Vec<clean::Impl>,
    //TODO:带泛型参数的impl块，但self是否该视为泛型？
    pub _generic_impl: Vec<clean::Impl>,
    pub _generic_impl_for_traits: Vec<clean::Impl>,
}

impl CrateImplCollection {
    pub fn new() -> Self {
        let impl_types = Vec::new();
        let impl_trait_for_types = Vec::new();
        let _generic_impl = Vec::new();
        let _generic_impl_for_traits = Vec::new();
        CrateImplCollection {
            impl_types,
            impl_trait_for_types,
            _generic_impl,
            _generic_impl_for_traits,
        }
    }

    pub fn add_impl(&mut self, impl_: &clean::Impl) {
        //println!("impl type = {:?}", impl_.for_);
        let _impl_type = &impl_.for_;
        //println!("impl type = {:?}", _impl_type);
        match impl_.trait_ {
            None => {
                //println!("No trait!");
                self.impl_types.push(impl_.clone());
            }
            Some(ref _ty_) => {
                //println!("trait={:?}", _ty_);
                self.impl_trait_for_types.push(impl_.clone());
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct FullNameMap {
    pub full_name_map: HashMap<DefId, (String, ItemType)>,
}

#[derive(Debug, Clone)]
pub struct TraitNameMap {
    pub trait_name_map: HashMap<DefId, String>,
}

#[derive(Debug, Clone)]
pub struct DefIdMap {
    pub full_name_map: FullNameMap,
    pub trait_name_map: TraitNameMap,
}

impl DefIdMap {
    pub fn new() -> Self {
        let full_name_map = FullNameMap::new();
        let trait_name_map = TraitNameMap::new();
        DefIdMap { full_name_map, trait_name_map }
    }
}

impl FullNameMap {
    pub fn new() -> Self {
        let full_name_map = HashMap::default();
        FullNameMap { full_name_map }
    }

    pub fn push_full_name(&mut self, def_id: &DefId, full_name: &String, item_type: ItemType) {
        self.full_name_map
            .insert(def_id.clone(), (full_name.clone(), item_type));
    }

    pub fn get_full_name(&self, def_id: &DefId) -> Option<String> {
        match self.full_name_map.get(def_id) {
            None => None,
            Some((full_name, _)) => Some(full_name.to_string()),
        }
    }
}

impl TraitNameMap {
    pub fn new() -> Self {
        let trait_name_map = HashMap::default();
        TraitNameMap { trait_name_map }
    }

    pub fn push_trait_name(&mut self, def_id: &DefId, trait_name: &String) {
        self.trait_name_map
            .insert(def_id.clone(), trait_name.clone());
    }

    pub fn get_trait_name(&self, def_id: &DefId) -> Option<String> {
        match self.trait_name_map.get(def_id) {
            None => None,
            Some(trait_name) => Some(trait_name.to_string()),
        }
    }
}


pub fn extract_impls_from_cache(
    cache: &Cache,
    tcx: &TyCtxt<'tcx>,
    def_id_map: &mut DefIdMap,
    mut api_graph: &mut ApiGraph,
) {
    let type_impl_maps = &cache.impls;
    let _trait_impl_maps = &cache.implementors;
    let paths = &cache.paths;

    let mut crate_impl_collection = CrateImplCollection::new();

    //construct the map of `did to type`
    for (did, (strings, item_type)) in paths {
        let full_name = full_path(&strings);
        def_id_map.full_name_map.push_full_name(&did, &full_name, *item_type);
        def_id_map.trait_name_map.push_trait_name(&did, &full_name);
    }

    let extertal_paths = &cache.external_paths;
    for (did, (strings, item_type)) in extertal_paths {
        let full_name = full_path(&strings);
        def_id_map.trait_name_map.push_trait_name(&did, &full_name);
        if prelude_type::is_preluded_struct(&full_name) 
        || did.is_local() 
        {
            def_id_map.full_name_map.push_full_name(&did, &full_name, *item_type);
        }
    }

    api_graph.set_full_name_map(&def_id_map.full_name_map);
    // println!("full name map: {:#?}", def_id_map.full_name_map);
    //首先提取所有type的impl
    //并查看是否实现Copy trait（后续语法检测中需要考量）
    for (did, impls) in type_impl_maps {
        //只添加可以在full_name_map中找到对应的did的type
        if def_id_map.full_name_map.get_full_name(did) != None {
            let full_name = def_id_map.full_name_map.get_full_name(did).unwrap().clone();
            let mut api_param = match api_graph.is_preluded_param(&full_name){
                Some(prelude_param) => { 
                    // println!("preluded parameter: {}", prelude_param.as_string());
                    prelude_param
                }
                None => {
                    // 先假设是struct，后续进行判断
                    println!("nonpreluded parameter: {}", full_name);
                    let api_struct = ApiStructure::new(full_name.to_string());
                    ApiParameter::Struct(api_struct)
                }
            };
            for impl_ in impls {
                // println!("structure full name {:?}", full_name);
                // println!("did: {:?}\nimpl info: {:#?}", did, impl_);
                // println!("impl name: {:#?}", impl_.impl_item.name);
                let mut base_generic_map = HashMap::new();
                match &*impl_.impl_item.kind {
                    ItemKind::ImplItem(impl_info) => {
                        // api_structure.set_clean_type(&impl_info.for_);
                        // println!("clean type of structure: {:#?}", impl_info.for_);
                        // println!("impl generics: {:#?}", impl_info.generics);
                        if let Some(trait_) = &impl_info.trait_ {
                            if trait_.whole_name().as_str() == "Copy" {
                                api_graph.copy_structs.push(full_name.to_string());
                            }
                            let trait_full_name = match def_id_map.trait_name_map.get_trait_name(&trait_.def_id()) {
                                Some(name) => name,
                                None => {
                                    let trait_name = tcx.def_path_str(trait_.def_id());
                                    def_id_map.trait_name_map.push_trait_name(&trait_.def_id(), &trait_name);
                                    trait_name
                                },
                            };
                            // println!("trait_: {:#?}", trait_);
                            // println!("trait whole name: {:#?}", trait_.whole_name());
                            // println!("trait def_id: {:#?}", trait_.def_id());
                            // println!("trait full name: {:?}", trait_full_name);
                            api_param.add_implemented_trait(&trait_full_name);
                        } else {
                            api_param.set_clean_type(&impl_info.for_);
                            api_param.judge_enum();
                            api_param.judge_generic();
                        }
                        base_generic_map = extract_generic_info(&tcx, &mut def_id_map.trait_name_map, &impl_info.generics, &base_generic_map, full_name.to_string());
                    }
                    _ => {}
                }
                _analyse_impl(&cache, &tcx, impl_.inner_impl(), def_id_map, &mut api_graph, base_generic_map, &mut api_param);
                crate_impl_collection.add_impl(impl_.inner_impl());
            }
            api_graph.add_api_parameter(api_param);
        }
    }
    //println!("analyse impl Type");
    //分析impl type类型
    // for impl_ in &crate_impl_collection.impl_types {
    //     //println!("analyse_impl_");
    //     _analyse_impl(&cache, &tcx, impl_, &full_name_map, &mut api_graph);
    // }

    // //println!("analyse impl Trait for Type");
    // for impl_ in &crate_impl_collection.impl_trait_for_types {
    //     println!("impl_trait_for_types_trait_: {:?}", impl_.trait_);
    //     _analyse_impl(&cache, &tcx, impl_, &full_name_map, &mut api_graph);
    // }
    //TODO：如何提取trait对应的impl，impl traitA for traitB? impl dyn traitA?下面的逻辑有误
    //for (did, impls) in trait_impl_maps {
    //   println!("trait:{:?}",did);
    //    //还是只看当前crate中的trait
    //    if full_name_map.get_full_name(did) != None {
    //        let full_name = full_name_map.get_full_name(did).unwrap();
    //        println!("full_name : {:?}", full_name);
    //        println!("{}", impls.len());
    //    }

    //}

    //println!("{:?}", crate_impl_collection);
}

pub fn extract_info_from_tcx(
    tcx: &TyCtxt<'tcx>,
    api_graph: &mut ApiGraph,
){
    let hir = tcx.hir();
    let ids = tcx.mir_keys(());
    let fn_ids: Vec<LocalDefId> = ids
        .clone()
        .into_iter()
        .filter(|id| {
            hir.body_owner_kind(hir.local_def_id_to_hir_id(*id))
                .is_fn_or_closure()
        })
        .collect();
    print!("Total function number: {}\n", fn_ids.len());
    for fn_id in &fn_ids {
        let fn_decl: &FnDecl<'_> = match hir.fn_decl_by_hir_id(hir.local_def_id_to_hir_id(*fn_id)){
            Some(value) => value,
            None => continue
        };
        
        // let output_info = collect_output_param_info(*fn_id, body);
        match api_graph.find_function_by_def_id(*fn_id){
            Ok(function) => {
                // println!("Function Name: {}", function.full_name);
                let body = tcx.optimized_mir(*fn_id);
                let unsafe_info = collect_unsafeconstruct_info(*fn_id, body);
                let rawptr_info = collect_rawptr_info(*fn_id, body);
                let drop_info = collect_drop_operation_info(*fn_id, body, tcx);
                let mutate_info = collect_mutate_operation_info(*fn_id, body);
                let return_info = collect_return_info(*fn_id, body);
                let output_type = collect_output_info(*fn_id, body);
                let func_types = collect_func_params_info(*fn_id, body);
                let input_types = collect_input_params_info(*fn_id, body, function.inputs.len());
                assert_eq!(input_types.len(), function.inputs.len());
                // let output_type = collect_output_param_info(*fn_id, body);
                // println!("{} Unsafe Info: {:#?}", function.full_name, unsafe_info);
                // println!("{} GetRawPtr Info: {:#?}", function.full_name, rawptr_info);
                // println!("{} Drop Info: {:#?}", function.full_name, drop_info);
                // println!("{} Mutate Info: {:#?}", function.full_name, mutate_info);
                // println!("{} Return Info: {:#?}", function.full_name, return_info);
                // println!("output type: {:#?}", output_type);
                // println!("Function {} Body: {:#?}", function.full_name, body);
                function.unsafe_info = unsafe_info;
                function.rawptr_info = rawptr_info;
                function.drop_info = drop_info;
                function.mutate_info = mutate_info;
                function.return_info = return_info;
                function.func_types = func_types;
                function.input_types = input_types;
                function.output_type = output_type;
            }
            Err(_) => (),
        }
        // println!("\nfn id: {:#?}\nFunction: {:#?}\n", fn_id, function);
    }
}

fn full_path(paths: &Vec<String>) -> String {
    let mut full = String::new();
    match paths.first() {
        None => {
            return full;
        }
        Some(path) => {
            full.push_str(path.as_str());
        }
    }
    let paths_num = paths.len();
    for i in 1..paths_num {
        let current_path = paths[i].as_str();
        full.push_str("::");
        full.push_str(current_path);
    }

    return full;
}

// 如果cache上有did对应的信息，就能获取到完整的方法或特性的名字
pub fn get_full_name(cache: &Cache, def_id: &DefId) -> Option<String> {
    let paths = &cache.paths;
    if let Some((strings, item_type)) = paths.get(def_id) {
        return Some(full_path(strings));
    }
    let external_paths = &cache.paths;
    if let Some((strings, item_type)) = external_paths.get(def_id) {
        return Some(full_path(strings));
    }
    None
}

pub fn _analyse_impl(
    cache: &Cache, 
    tcx: &TyCtxt<'tcx>, 
    impl_: &clean::Impl, 
    def_id_map: &mut DefIdMap,
    api_graph: &mut ApiGraph,
    base_generic_map: HashMap<String, Vec<String>>,
    api_parameter: &mut ApiParameter,
) {
    let inner_items = &impl_.items;

    //BUG FIX: TRAIT作为全限定名只能用于输入类型中带有self type的情况，这样可以推测self type，否则需要用具体的类型名

    let trait_full_name = match &impl_.trait_ {
        None => None,
        Some(trait_) => {
            let trait_ty_def_id = &trait_.def_id();
            // println!("def_id: {:#?}", trait_ty_def_id);
            let trait_full_name = def_id_map.full_name_map.get_full_name(trait_ty_def_id);
            if let Some(trait_name) = trait_full_name {
                Some(trait_name.clone())
            } else {
                None
            }
        }
    };
    // println!("trait_full_name: {:#?}",trait_full_name);

    let impl_ty_def_id = &impl_.for_.def_id_no_primitives(); // &impl_.for_.def_id();
    let type_full_name = if let Some(def_id) = impl_ty_def_id {
        let type_name = def_id_map.full_name_map.get_full_name(def_id);
        if let Some(real_type_name) = type_name {
            Some(real_type_name.clone())
        } else {
            None
        }
    } else {
        None
    };
    let mut count = 0;
    for item in inner_items {
        //println!("item_name, {:?}", item.name.as_ref().unwrap());
        // println!("item: {:#?}", item);
        let def_id = item.def_id;
        match &*item.kind {
            //TODO:这段代码暂时没用了，impl块里面的是method item，而不是function item,暂时留着，看里面是否会出现function item
            clean::FunctionItem(_function) => {
                let function_name = String::new();
                //使用全限定名称：type::f
                //function_name.push_str(type_full_name.as_str());
                //function_name.push_str("::");
                //function_name.push_str(item.name.as_ref().unwrap().as_str());
                // println!("function name in impl:{:?}", function_name);
            }
            // impl Struct {..} Method
            clean::MethodItem(_method, ..) => {
                // println!("clean method generics: {:#?}", _method.generics);
                count += 1;
                let decl = _method.decl.clone();
                let clean::FnDecl { inputs, output, .. } = decl;
                let mut inputs = api_util::_extract_input_types(&inputs);
                let output = api_util::_extract_output_type(&output);
                //println!("input types = {:?}", inputs);

                let mut contains_self_type = false;

                let input_len = inputs.len();
                for index in 0..input_len {
                    let ty_ = &inputs[index];
                    if is_param_self_type(ty_) {
                        contains_self_type = true;
                        let raplaced_ty = replace_self_type(ty_, &impl_.for_);
                        inputs[index] = raplaced_ty;
                    }
                }
                //println!("after replace, input = {:?}", inputs);

                let output = match output {
                    None => None,
                    Some(ty_) => {
                        if is_param_self_type(&ty_) {
                            let replaced_type = replace_self_type(&ty_, &impl_.for_);
                            Some(replaced_type)
                        } else {
                            Some(ty_)
                        }
                    }
                };

                let mut method_name = String::new();
                //使用全限定名称：type::f
                //如果函数输入参数中含有self type，则使用trait name（也可以使用type name）
                //如果函数输入参数中不含有self type，则使用type name
                let method_type_name = if contains_self_type {
                    if let Some(ref trait_name) = trait_full_name {
                        trait_name.clone()
                    } else if let Some(ref type_name) = type_full_name {
                        type_name.clone()
                    } else {
                        //println!("trait not in current crate.");
                        //println!("type not in current crate.");
                        return;
                    }
                } else {
                    if let Some(ref type_name) = type_full_name {
                        type_name.clone()
                    } else {
                        //println!("type not in current crate.");
                        return;
                    }
                };
                method_name.push_str(method_type_name.as_str());
                method_name.push_str("::");
                method_name.push_str(&item.name.as_ref().unwrap().as_str());
                //println!("method name in impl:{:?}", method_name);
                println!("\n>>>> extract method {} generic info", method_name);
                let generics = extract_generic_info(&tcx, &mut def_id_map.trait_name_map, &_method.generics, &base_generic_map, method_name.clone());
                println!("extract generic info finished <<<<\n");
                let api_unsafety = ApiUnsafety::_get_unsafety_from_fnheader(&_method.header);
                //生成api function
                //如果是实现了trait的话，需要把trait的全路径也包括进去
                let api_function = match &impl_.trait_ {
                    None => {
                        // structure implmented method
                        // println!("method_full_name {}", method_name);
                        ApiFunction {
                            full_name: method_name.clone(),
                            generics,
                            inputs,
                            output,
                            _trait_full_path: None,
                            _unsafe_tag: api_unsafety,
                            def_id,
                            unsafe_info: HashMap::new(),
                            rawptr_info: HashMap::new(),
                            drop_info: HashMap::new(),
                            mutate_info: HashMap::new(),
                            return_info: HashMap::new(),
                            func_types: HashSet::new(),
                            input_types: Vec::new(),
                            output_type: None,
                            need_functions: HashMap::new(),
                            next_functions: HashSet::new(),
                            weight: 0,
                        }
                    },
                    Some(_) => {
                        // 暂时不要from<T>
                        // 排除fmt, clone
                        // if method_name.as_str() == "from" {
                        //     if !generics.is_empty() {
                        //         continue;
                        //     }
                        // }
                        println!("trait name: {}", method_name);
                        if method_name.as_str() == "core::fmt::Debug::fmt"
                        || method_name.as_str() == "core::clone::Clone::clone" {
                            continue;
                        }
                        // trait funciton
                        if let Some(ref real_trait_name) = trait_full_name {
                            // println!("trait_full_name {}", method_name);
                            ApiFunction {
                                full_name: method_name.clone(),
                                generics,
                                inputs,
                                output,
                                _trait_full_path: Some(real_trait_name.clone()),
                                _unsafe_tag: api_unsafety,
                                def_id,
                                unsafe_info: HashMap::new(),
                                rawptr_info: HashMap::new(),
                                drop_info: HashMap::new(),
                                mutate_info: HashMap::new(),
                                return_info: HashMap::new(),
                                func_types: HashSet::new(),
                                input_types: Vec::new(),
                                output_type: None,
                                need_functions: HashMap::new(),
                                next_functions: HashSet::new(),
                                weight: 0,
                            }
                        } else {
                            // println!("Trait not found in current crate.");
                            return;
                        }
                    }
                };
                api_parameter.add_implemented_function(&method_name);
                api_graph.add_api_function(api_function);
            }
            clean::StructItem(_struct) => {
                println!("struct item: {:#?}", _struct);
            }
            _ => {
                // println!("uncovered item: {:#?}", item);
                println!("No covered item {:#?}\nNo covered item kind: {:#?}\n", &item.name, &item.kind);
            }
        }
    }
    // print!("Total inner items number: {}\n", count);
}

//递归判断一个参数是否是self类型的
//TODO：考虑在resolved path里面的括号里面可能存在self type
fn is_param_self_type(ty_: &clean::Type) -> bool {
    if ty_.is_self_type() {
        return true;
    }
    match ty_ {
        clean::Type::BorrowedRef { type_, .. } => {
            let inner_ty = &**type_;
            if is_param_self_type(inner_ty) {
                return true;
            } else {
                return false;
            }
        }
        clean::Type::Path { path, .. } => {
            let segments = &path.segments;
            for path_segment in segments {
                let generic_args = &path_segment.args;
                match generic_args {
                    clean::GenericArgs::AngleBracketed { args, .. } => {
                        for generic_arg in args {
                            if let clean::GenericArg::Type(generic_ty) = generic_arg {
                                if is_param_self_type(generic_ty) {
                                    return true;
                                }
                            }
                        }
                    }
                    clean::GenericArgs::Parenthesized { inputs, output } => {
                        for input_type in inputs {
                            if is_param_self_type(input_type) {
                                return true;
                            }
                        }
                        if let Some(output_type) = output {
                            if is_param_self_type(output_type) {
                                return true;
                            }
                        }
                    }
                }
            }
            return false;
        }
        _ => {
            return false;
        }
    }
}

//将self类型替换为相应的结构体类型
pub fn replace_self_type(self_type: &clean::Type, impl_type: &clean::Type) -> clean::Type {
    if self_type.is_self_type() {
        return impl_type.clone();
    }
    match self_type {
        clean::Type::BorrowedRef {
            lifetime,
            mutability,
            type_,
        } => {
            let inner_type = &**type_;
            if is_param_self_type(inner_type) {
                let replaced_type = replace_self_type(inner_type, impl_type);
                return clean::Type::BorrowedRef {
                    lifetime: lifetime.clone(),
                    mutability: mutability.clone(),
                    type_: Box::new(replaced_type),
                };
            } else {
                return self_type.clone();
            }
        }
        clean::Type::Path { path } => {
            //, is_generic} => {
            if !is_param_self_type(self_type) {
                return self_type.clone();
            }
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
                                if is_param_self_type(generic_type) {
                                    let replaced_type = replace_self_type(generic_type, impl_type);
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
                            if is_param_self_type(input_type) {
                                let replaced_type = replace_self_type(input_type, impl_type);
                                new_inputs.push(replaced_type);
                            } else {
                                new_inputs.push(input_type.clone());
                            }
                        }
                        let new_output = match output {
                            None => None,
                            Some(output_type) => {
                                let new_output_type = if is_param_self_type(output_type) {
                                    let replaced_type =
                                        Box::new(replace_self_type(output_type, impl_type));
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
        }
        _ => {
            return self_type.clone();
        }
    }
}

pub fn facilitate_type_name(type_name: &String) -> String {
    // 去option和Result
    // std::result::Result<Type, Error>
    // std::option::Option<Type>
    let mut simplified_type_name = type_name.clone();
    let name_len = type_name.len();
    // 根据前缀处理
    if simplified_type_name.starts_with("std::option::Option<") {
        simplified_type_name = facilitate_type_name(&simplified_type_name.drain(20..name_len-1).collect());
    } else if simplified_type_name.starts_with("std::result::Result<"){
        let result_type_name = facilitate_type_name(&simplified_type_name.drain(20..name_len-1).collect());
        // 处理Error，即空格
        let names: Vec<String> = result_type_name.split(" ").map(
            |name| name.to_string()
        ).collect();
        simplified_type_name = names[0].clone();
    } else if simplified_type_name.starts_with("&mut") {
        simplified_type_name = type_name.chars().skip(5).collect();
        // ty2 = ty2.as_str().split(' ').collect::<B>()[1];
    } else if simplified_type_name.starts_with('&') {
        simplified_type_name = type_name.chars().skip(1).collect();
    } else if simplified_type_name.starts_with("*const"){
        simplified_type_name = type_name.chars().skip(7).collect();
    } else if simplified_type_name.starts_with("*mut") {
        simplified_type_name = type_name.chars().skip(5).collect();
    } else if simplified_type_name.starts_with('*'){
        simplified_type_name = type_name.chars().skip(1).collect();
    } else {
        simplified_type_name = type_name.to_string();
    }

    simplified_type_name
}

pub fn get_inner_type(type_: &clean::Type) -> clean::Type {
    match type_ {
        clean::Type::Tuple(vec_types) => {
            if vec_types.len() == 0 {
                return type_.clone()
            }
            return get_inner_type(&vec_types[0]);
        }
        clean::Type::Slice(box_type) => get_inner_type(&*box_type),
        clean::Type::Array(box_type, s) => get_inner_type(&*box_type),
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

// 用于分析是否冲突
pub fn lifetime_analyse(
    move_lifetime: &HashMap<usize, usize>, 
    mut_ref_lifetime: &HashMap<usize, Vec<(usize, usize)>>,
    immut_ref_lifetime: &HashMap<usize, Vec<(usize, usize)>>,
) -> bool
{
    // move 分析
    for (move_start_index, move_end_index) in move_lifetime {
        if let Some(mut_lifetimes) = mut_ref_lifetime.get(&move_start_index) {
            for (mut_start_index, mut_end_index) in mut_lifetimes{
                if mut_end_index >= move_end_index {
                    return false;
                }
            }
        }
        if let Some(immut_lifetimes) = immut_ref_lifetime.get(&move_start_index) {
            for (immut_start_index, immut_end_index) in immut_lifetimes {
                if immut_end_index >= move_end_index {
                    return false;
                }
            }
        }
    }
    // ref 分析
    for (create_index, mut_lifetimes) in mut_ref_lifetime {
        let (mut last_mut_start_index, mut last_mut_end_index) = (0, 0);
        for (mut_start_index, mut_end_index) in mut_lifetimes{
            if let Some(immut_lifetimes) = immut_ref_lifetime.get(&create_index) {
                for (immut_start_index, immut_end_index) in immut_lifetimes {
                    if immut_start_index <= mut_end_index && mut_start_index <= immut_end_index {
                        return false;
                    }
                }
            }
            if &last_mut_start_index <= mut_end_index && mut_start_index <= &last_mut_end_index {
                return false;
            }
            (last_mut_start_index, last_mut_end_index) = (*mut_start_index, *mut_end_index);
        }
    }
    true
}

// 用于更新存活的生命周期以避免语法错误
// 包括mut和immut
pub fn update_lifetime(
    relation_map: &HashMap<usize, Vec<(usize, bool)>>, 
    mut_ref_lifetime: &mut HashMap<usize, Vec<(usize, usize)>>,
    immut_ref_lifetime: &mut HashMap<usize, Vec<(usize, usize)>>,
    update_index: usize,
    end_index: usize,
)
{
    // if let Some(mut_lifetimes) = mut_ref_lifetime.get_mut(update_index) {
        
    // } else {
        
    // }

    if let Some(relations) = relation_map.get(&update_index) {
        for (called_index, is_mut) in relations {
            if *is_mut {
                let mut push_flag = true;
                if let Some(mut_lifetimes) = mut_ref_lifetime.get_mut(called_index) {
                    for (mut_start_index, mut_end_index) in mut_lifetimes {
                        if *mut_start_index == update_index {
                            *mut_end_index = end_index;
                            push_flag = false;
                            break;
                        }
                    }
                } else {
                    push_flag = false;
                    mut_ref_lifetime.insert(*called_index, vec![(update_index, end_index)]);
                }
                if push_flag {
                    if let Some(mut_lifetimes) = mut_ref_lifetime.get_mut(called_index) {
                        mut_lifetimes.push((update_index, end_index));
                    }
                }
            } else {
                let mut push_flag = true;
                if let Some(immut_lifetimes) = immut_ref_lifetime.get_mut(called_index) {
                    for lifetime in immut_lifetimes {
                        if lifetime.0 == update_index {
                            lifetime.1 = end_index;
                            push_flag = false;
                            break;
                        }
                    }
                } else {
                    push_flag = false;
                    immut_ref_lifetime.insert(*called_index, vec![(update_index, end_index)]);
                }
                if push_flag {
                    if let Some(immut_lifetimes) = immut_ref_lifetime.get_mut(called_index) {
                        immut_lifetimes.push((update_index, end_index));
                    }
                }
            }
            update_lifetime(&relation_map, mut_ref_lifetime, immut_ref_lifetime, *called_index, end_index);
        }
    }
}

// 处理泛型数据
// 用于从clean::Generic数据中收集ApiFunction所需的泛型信息
pub fn extract_generic_info(
    tcx: &TyCtxt<'tcx>, 
    trait_name_map: &mut TraitNameMap,
    generics: &clean::Generics,
    base_generic_map: &HashMap<String, Vec<String>>,
    base_name: String,
) -> HashMap<String, Vec<String>> {
    // TODO: from<T>, clone, copy, debug, display不要加入
    if base_name == String::from("ordnung::RawOccupiedEntryMut") {
        println!("debug generics info: {:#?}", generics);
    }
    if base_name == String::from("ordnung::RawOccupiedEntryMut::get_mut") {
        println!("debug generics info: {:#?}", generics);
        println!("debug origin generics map: {:#?}", base_generic_map);
    }
    println!("base name: {}", base_name);
    // println!("generic info: {:#?}", generics);
    let mut generic_map = base_generic_map.clone();
    for i in 0..generics.params.len() {
        let mut type_name = base_name.clone();
        type_name.push_str("::");
        type_name.push_str(generics.params[i].name.as_str());
        if !generics.params[i].kind.is_type() { continue; }
        // if generics.where_predicates.len() <= i
        if let Some(bounds) = generics.params[i].get_bounds() {
            let mut res_bounds = Vec::new();
            for bound in bounds {
                if let Some(trait_) = bound.get_trait_path() {
                    let trait_full_name = match trait_name_map.get_trait_name(&trait_.def_id()) {
                        Some(name) => name,
                        None => {
                            let trait_name = tcx.def_path_str(trait_.def_id());
                            trait_name_map.push_trait_name(&trait_.def_id(), &trait_name);
                            trait_name
                        },
                    };
                    res_bounds.push(trait_full_name);
                }
            }
            if let Some(bounds_) = generic_map.get(&type_name) {
                println!("Error when generic bounding");
            } else {
                generic_map.insert(type_name.clone(), res_bounds);
            }
        }
    }
    // println!("predicating");
    for where_predicate in &generics.where_predicates {
        match where_predicate {
            clean::WherePredicate::BoundPredicate{ ty, bounds, bound_params } => {
                match &ty {
                    clean::Type::Generic(symbol_) => {
                        let mut type_name = base_name.clone();
                        type_name.push_str("::");
                        type_name.push_str(symbol_.as_str());
                        // let type_name = symbol_.as_str().to_string();
                        if let Some(bounds) = where_predicate.get_bounds() {
                            // println!("generic bouds: {:#?}", bounds);
                            let mut predicate_bounds = Vec::new();
                            for bound in bounds {
                                // println!("bound: {:#?}", bound);
                                if let Some(trait_) = bound.get_trait_path() {
                                    let mut trait_full_name = match trait_name_map.get_trait_name(&trait_.def_id()) {
                                        Some(name) => name,
                                        None => {
                                            let trait_name = tcx.def_path_str(trait_.def_id());
                                            trait_name_map.push_trait_name(&trait_.def_id(), &trait_name);
                                            trait_name
                                        },
                                    };
                                    if trait_full_name == "core::convert::Into" {
                                        trait_full_name = String::from("Into not implemented for now");
                                    }
                                    predicate_bounds.push(trait_full_name);
                                }
                            }
                            if let Some(bounds_) = generic_map.get_mut(&type_name) {
                                bounds_.append(&mut predicate_bounds)
                            } else {
                                generic_map.insert(type_name.clone(), predicate_bounds);
                            }
                        }
                    },
                    clean::Type::Path { path } => {
                        for (type_name, bounds) in &mut generic_map {
                            bounds.push(String::from("path can't solve"));
                        }
                    },
                    _ =>{
                        for (type_name, bounds) in &mut generic_map {
                            bounds.push(String::from("other ty can't solve"));
                        }
                    }
                }
            }
            _ => { }
        }
    }
    // if base_name == String::from("ordnung::RawOccupiedEntryMut") {
    //     println!("debug generics map: {:#?}", generic_map);
    // }
    // println!("map: {:#?}", generic_map);
    generic_map
}

pub fn extract_data_from_ctxt(
    ctxt: &mut DocContext<'tcx>,
    def_id_map: &mut DefIdMap,
    api_graph: &mut ApiGraph,
) {
    extract_impls_from_cache(&ctxt.cache, &ctxt.tcx, def_id_map, api_graph);
    extract_info_from_tcx(&ctxt.tcx, api_graph);
    // let hir_map = ctxt.tcx.hir();
    // let impls = &ctxt.cache.impls.clone();
    // let ids = impls.keys().clone();
    // for did in ids {
    //     //只添加可以在full_name_map中找到对应的did的type
    //     if def_id_map.full_name_map.get_full_name(did) != None {
    //         let full_name = def_id_map.full_name_map.get_full_name(did).unwrap();
    //         // println!("full name is: {:#?}", full_name);
    //         if let Some(local_did) = did.as_local() {
    //             match hir_map.get(hir_map.local_def_id_to_hir_id(local_did)) {
    //                 Node::Ty(ty_) => {
    //                     // println!("Successful!\nfn_id: {:#?}\nNode: {:#?}", local_did, hir_map.get(hir_map.local_def_id_to_hir_id(local_did)));
    //                     let clean_type = clean::clean_qpath(&ty_, ctxt);
    //                     // println!("clean type: {:#?}", clean_type);
    //                 },
    //                 Node::Item(item) => {
    //                     // println!("Successful!\nfn_id: {:#?}\nNode: {:#?}", local_did, hir_map.get(hir_map.local_def_id_to_hir_id(local_did)));
    //                     match &item.kind {
    //                         rustc_hir::ItemKind::Struct(data, ..) => {
    //                             for field in data.fields(){
    //                                 let clean_type = clean::clean_qpath(field.ty, ctxt);
    //                                 // println!("clean type: {:#?}", clean_type);
    //                             }
    //                         },
    //                         rustc_hir::ItemKind::Enum(data, ..) => {
                                
    //                         },
    //                         _ => {
    //                             println!("Mismatch item kind: {:#?}", item.kind);
    //                         }
    //                     }
    //                 },
    //                 _ => {
    //                     println!("Failed!\nfn_id: {:#?}\nNode: {:#?}", local_did, hir_map.get(hir_map.local_def_id_to_hir_id(local_did)));
    //                 }
    //             };
    //         } else {
    //             println!("did to local_didi failed: {:#?}", did);
    //         }
    //     }
    // }
}