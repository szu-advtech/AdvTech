use crate::fuzz_target::dataflow::*;
use crate::fuzz_target::tracker::{Tracker, TrackerState};
use crate::fuzz_target::def_use::DefUseAnalysis;
use crate::fuzz_target::operation_sequence::*;
use crate::fuzz_target::operation_sequence::{OperationSequenceInfo, StatementId, StatementInfo};
use crate::fuzz_target::impl_util::*;

use rustc_hir::def_id::LocalDefId;
use rustc_middle::ty::{self, TyCtxt, FnDef};
use rustc_span::def_id::DefId;
use rustc_middle::mir::visit::{
    MutatingUseContext, NonMutatingUseContext, NonUseContext, PlaceContext,
};
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs;
use rustc_middle::mir::{StatementKind, Body, Local, LocalInfo, Place, ProjectionElem, Rvalue, ClearCrossCrate, Safety, Operand, Mutability, ImplicitSelfKind};
use rustc_middle::mir::terminator::TerminatorKind;
use rustc_ast;
use rustdoc::error::Error;
use rustdoc::clean;
use std::collections::{HashMap, HashSet};
use rustc_span::Span;
use lazy_static::*;
use std::sync::Mutex;

lazy_static! {
    static ref DEF_IDS: Mutex<HashSet<DefId>> = {
        Mutex::new(HashSet::new())
    };
}

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

pub fn collect_unsafeconstruct_info(
    fn_id: LocalDefId,
    body: &Body<'_>
) -> HashMap<StatementId, StatementInfo> {
    let mut unsafe_construct: HashMap<StatementId, StatementInfo> = HashMap::new();
    // find unsafe construct
    for each_source_scope_data in &body.source_scopes {
        if let ClearCrossCrate::Set(source_scope_local_data) = &each_source_scope_data.local_data  {
            // println!("each_source_scope_data: {:#?}", each_source_scope_data);
            match source_scope_local_data.safety {
                Safety::Safe => {},
                Safety::ExplicitUnsafe(hir_id) => {
                    // TODO:使用hir_id获取unsafe block块的信息（里面涉及到的struct type）
                    // println!("Explicit Unsafe: {:#?}", hir_id);
                    // println!("unsafe basic_blocks: {:#?}", body.basic_blocks());

                    // 在mir层，unsafe块无法直接找到，通过span信息查找
                    // let Span{base_or_index: span_start, len_or_tag: len, ctxt_or_zero: span_end} = each_source_scope_data.span;
                    // println!("test...:{},{},{}",span_start, len, span_end);
                    for each_basicblock in body.basic_blocks() {
                        println!("bb: {:#?}", each_basicblock);
                        for each_statement in &each_basicblock.statements {
                            // each_source_scope_data 所指向的span即是unsafe block的span
                            // 获取该unsafe block下的place
                            println!("statement: {:?}\nkind: {:?}", each_statement, each_statement.kind);
                            if each_source_scope_data.span.contains(each_statement.source_info.span) {
                                // println!("statement content: {:#?}", each_statement);
                                // println!("statement span: {:#?}", each_statement.source_info.span);
                                // println!("statement scope: {:#?}", each_statement.source_info.scope);
                                match &each_statement.kind {
                                    StatementKind::Assign(statebox) => {
                                        let place = &statebox.0;
                                        println!("local: {:?}, projection: {:?}", place.local, place.projection);
                                        let unsafe_id = StatementId::new(fn_id, place.local);
                                        let unsafe_info = StatementInfo {
                                            type_name: (OperationType::UnsafeBlock, body.local_decls[place.local].ty.to_string()),
                                            src: None,
                                            span: body.local_decls[place.local].source_info.span,
                                            gen_bbs: Vec::new(),
                                            kill_bbs: Vec::new(),
                                        };
                                        // println!("unsafe construct insert {:#?}, {:#?}", unsafe_id, unsafe_info);
                                        unsafe_construct.insert(unsafe_id, unsafe_info);
                                    },
                                    StatementKind::FakeRead(statebox) => {},
                                    StatementKind::SetDiscriminant{place, variant_index} => {},
                                    StatementKind::StorageLive(local) | StatementKind::StorageDead(local) => {
                                        
                                    },
                                    _ => {},
                                }
                            }
                            
                        }
                    }
                    // println!("Source Scope Data: {:#?}", each_source_scope_data);
                    // unsafe_construct.insert(each_source_scope_data.span, source_scope_local_data.safety);
                }, 
                Safety::FnUnsafe => {
                    // println!("Fn Unsafe API");
                    // 将所有参数的API都加入列表中
                    // unsafe_construct.insert(each_source_scope_data.span, source_scope_local_data.safety);
                    for each_basicblock in body.basic_blocks() {
                        for each_statement in &each_basicblock.statements {
                            // each_source_scope_data 所指向的span即是unsafe block的span
                            // 获取该unsafe block下的place
                            // if each_source_scope_data.span.contains(each_statement.source_info.span) {
                            // println!("statement content: {:#?}", each_statement);
                            // println!("statement span: {:#?}", each_statement.source_info.span);
                            // println!("statement scope: {:#?}", each_statement.source_info.scope);
                            match &each_statement.kind {
                                StatementKind::Assign(statebox) => {
                                    let place = &statebox.0;
                                    let unsafe_id = StatementId::new(fn_id, place.local);
                                    let unsafe_info = StatementInfo {
                                        type_name: (OperationType::UnsafeBlock, body.local_decls[place.local].ty.to_string()),
                                        src: None,
                                        span: body.local_decls[place.local].source_info.span,
                                        gen_bbs: Vec::new(),
                                        kill_bbs: Vec::new(),
                                    };
                                    // println!("unsafe construct insert {:#?}, {:#?}", unsafe_id, unsafe_info);
                                    unsafe_construct.insert(unsafe_id, unsafe_info);
                                },
                                StatementKind::FakeRead(statebox) => {},
                                StatementKind::SetDiscriminant{place, variant_index} => {},
                                StatementKind::StorageLive(local) | StatementKind::StorageDead(local) => {
                                    
                                },
                                _ => {},
                            }
                            // }
                            
                        }
                    }
                }
                Safety::BuiltinUnsafe => {
                    // println!("Built in Unsafe API", );
                    // println!("Fn Unsafe API");
                    // 将所有参数的API都加入列表中
                    // unsafe_construct.insert(each_source_scope_data.span, source_scope_local_data.safety);
                    for each_basicblock in body.basic_blocks() {
                        println!("bb: {:#?}", each_basicblock);
                        for each_statement in &each_basicblock.statements {
                            // each_source_scope_data 所指向的span即是unsafe block的span
                            // 获取该unsafe block下的place
                            // if each_source_scope_data.span.contains(each_statement.source_info.span) {
                            // println!("statement content: {:#?}", each_statement);
                            // println!("statement span: {:#?}", each_statement.source_info.span);
                            // println!("statement scope: {:#?}", each_statement.source_info.scope);
                            match &each_statement.kind {
                                StatementKind::Assign(statebox) => {
                                    let place = &statebox.0;
                                    let unsafe_id = StatementId::new(fn_id, place.local);
                                    let unsafe_info = StatementInfo {
                                        type_name: (OperationType::UnsafeBlock, body.local_decls[place.local].ty.to_string()),
                                        src: None,
                                        span: body.local_decls[place.local].source_info.span,
                                        gen_bbs: Vec::new(),
                                        kill_bbs: Vec::new(),
                                    };
                                    // println!("unsafe construct insert {:#?}, {:#?}", unsafe_id, unsafe_info);
                                    unsafe_construct.insert(unsafe_id, unsafe_info);
                                },
                                StatementKind::FakeRead(statebox) => {},
                                StatementKind::SetDiscriminant{place, variant_index} => {},
                                StatementKind::StorageLive(local) | StatementKind::StorageDead(local) => {
                                    
                                },
                                _ => {},
                            }
                            // }
                            
                        }
                    }
                }
            }
        }
    }
    let mut def_use_analysis = DefUseAnalysis::new(body);
    def_use_analysis.analyze(body);
    unsafe_construct = collect_operation_src_info(unsafe_construct, body, &def_use_analysis);
    unsafe_construct = collect_gen_kill_bbs(unsafe_construct, body, &def_use_analysis);
    // println!("after unsafe construct: {:#?}", unsafe_construct);
    unsafe_construct
}

pub fn collect_rawptr_info(
    fn_id: LocalDefId,
    body: &Body<'_>,
) -> HashMap<StatementId, StatementInfo> {
    // println!("Return Lifetime: ");
    // let return_ty = &body.return_ty();
    // match return_ty.kind() {
    //     ty::TyKind::Ref(region, ty, mutability) => {
    //         println!("return type: {:?}\nregion: {:#?}", return_ty, region);
    //     }
    //     _ => {}
    // }

    let mut rawptrs: HashMap<StatementId, StatementInfo> = HashMap::new();
    println!("Collect GetRawPtr Info:");
    // find raw pointer
    for each_basicblock in body.basic_blocks() {
        // println!("basic block: {:#?}", each_basicblock);
        for each_statement in &each_basicblock.statements {
            if let StatementKind::Assign(statebox) = &each_statement.kind {
                //println!("State 0 = {:?}", statebox.0); // Type: rustc_middle::mir::Place
                //println!("State 1 = {:?}", statebox.1); // Type: rustc_middle::mir::Rvalue
                if let Rvalue::AddressOf(mutability, place) = &statebox.1 {
                    //println!("{:?}, {:?}, {:?}, statement = {:?}", mutability, place.local, body.local_decls[place.local], statebox.1);
                    let rawptr_id = StatementId::new(fn_id, place.local);
                    let rawptr_info = StatementInfo {
                        type_name: (OperationType::GetRawPtr, body.local_decls[place.local].ty.to_string()),
                        src: None,
                        span: body.local_decls[place.local].source_info.span,
                        gen_bbs: Vec::new(),
                        kill_bbs: Vec::new(),
                    };
                    rawptrs.insert(rawptr_id, rawptr_info);
                }
            }
            // 看看生命周期
        }
    }

    let mut def_use_analysis = DefUseAnalysis::new(body);
    def_use_analysis.analyze(body);

    let rawptrs = collect_operation_src_info(rawptrs, body, &def_use_analysis);
    collect_gen_kill_bbs(rawptrs, body, &def_use_analysis)
}

pub fn collect_output_info (
    fn_id: LocalDefId,
    body: &Body<'_>,
) -> Option<String> {
    let mut return_operations: HashMap<StatementId, StatementInfo> = HashMap::new();
    println!("Collect Output Info:");
    for each_basicblock in body.basic_blocks() {
        // println!("bb: {:#?}", each_basicblock);
        if let Some(terminator) = &each_basicblock.terminator {
            match &terminator.kind {
                TerminatorKind::Return => {
                    let drop_operation_id = StatementId::new(fn_id, Local::from_usize(0));
                    let drop_operation_info = StatementInfo {
                        type_name: (OperationType::Drop, body.local_decls[Local::from_usize(0)].ty.to_string()),
                        src: None,
                        span: terminator.source_info.span, // use body.local_decls[tkd_place.local].source_info.span to get the source local
                        gen_bbs: Vec::new(),
                        kill_bbs: Vec::new(),
                    };
                    return_operations.insert(drop_operation_id, drop_operation_info);
                },
                _ => {},
            }
        }
    }

    let mut def_use_analysis = DefUseAnalysis::new(body);
    def_use_analysis.analyze(body);
    println!("return track!");
    let return_operations = collect_operation_src_info(return_operations, body, &def_use_analysis);
    // println!("return info: {:#?}", return_operations);
    for (id, info) in &return_operations {
        match &info.src {
            Some(StatememtSrc::ParamSrc(param_src_context)) => {
                let type_name = param_src_context.struct_type.clone();
                return Some(type_name);
            },
            _ => {}
        }
    }
    None
}

pub fn collect_drop_operation_info(
    fn_id: LocalDefId,
    body: &Body<'_>,
    tcx: &TyCtxt<'tcx>,
) -> HashMap<StatementId, StatementInfo> {
    let mut drop_operations: HashMap<StatementId, StatementInfo> = HashMap::new();
    // TODO:find related basic block statement
    println!("Collect Drop Info:");
    // find drop_operation
    for each_basicblock in body.basic_blocks() {
        // println!("bb:\n{:#?}", each_basicblock);
        // println!("termiantor kind:\n{:?}", each_basicblock.terminator);
        if let Some(terminator) = &each_basicblock.terminator {
            match &terminator.kind {
                TerminatorKind::Drop{ place: tkd_place, target: tkd_target, unwind: tkd_unwind } => {
                    // 存在Drop行为，判断Drop的对象，如果为输入或者返回值，则添加为Drop信息
                    println!("Drop Catch in Terminator");
                    // 这里添加Drop信息收集
                    let drop_operation_id_t = StatementId::new(fn_id, tkd_place.local);
                    let drop_operation_info_t = StatementInfo {
                        type_name: (OperationType::Drop, body.local_decls[tkd_place.local].ty.to_string()),
                        src: None,
                        span: terminator.source_info.span, // use body.local_decls[tkd_place.local].source_info.span to get the source local
                        gen_bbs: Vec::new(),
                        kill_bbs: Vec::new(),
                    };
                    drop_operations.insert(drop_operation_id_t, drop_operation_info_t);

                    for last_statement_in_bb in &each_basicblock.statements {
                        if let StatementKind::StorageDead(storagedeallocal) = last_statement_in_bb.kind {
                            println!("Drop Catch in Statement \n{:#?}", last_statement_in_bb);
                            // println!("### {:?}", last_statement_in_bb.source_info);
                            let drop_operation_id = StatementId::new(fn_id, storagedeallocal);
                            let drop_operation_info = StatementInfo {
                                type_name: (OperationType::Drop, body.local_decls[storagedeallocal].ty.to_string()),
                                src: None,
                                span: terminator.source_info.span, // use body.local_decls[tkd_place.local].source_info.span to get the source local
                                gen_bbs: Vec::new(),
                                kill_bbs: Vec::new(),
                            };
                            drop_operations.insert(drop_operation_id, drop_operation_info);
                        }
                    }
                },
                TerminatorKind::DropAndReplace{..} | TerminatorKind::GeneratorDrop => {
                    println!("DropAndReplace or GeneratorDrop");
                },
                TerminatorKind::Call{ func, args, destination, cleanup, from_hir_call, fn_span } => {
                    println!("Call Track for {:#?}", terminator.kind);
                    // println!("{:#?}", func);
                    match func {
                        Operand::Constant(value) => {
                            if let Some(const_) = func.constant() {
                                println!("Span: {:#?}", value.span);
                                let const_ty = const_.literal.ty();
                                if let FnDef(def_id, substs) = *const_ty.kind() {
                                    if tcx.is_mir_available(def_id) {
                                        // let mut def_ids_set = DEF_IDS.lock().unwrap();
                                        // if !def_ids_set.contains(&def_id) {
                                        let fn_body = tcx.optimized_mir(def_id);
                                    }
                                }
                            }
                        },
                        _ => {},
                    }
                },
                _ => {}
            }
        }
    }

    let mut def_use_analysis = DefUseAnalysis::new(body);
    def_use_analysis.analyze(body);
    let drop_operations = collect_operation_src_info(drop_operations, body, &def_use_analysis);
    collect_gen_kill_bbs(drop_operations, body, &def_use_analysis)
}

pub fn collect_mutate_operation_info(
    fn_id: LocalDefId,
    body: &Body<'_>,
) -> HashMap<StatementId, StatementInfo> {
    let mut drop_operations: HashMap<StatementId, StatementInfo> = HashMap::new();
    // TODO:find related basic block statement
    println!("Collect Mutate Info:");
    // find mutate_operation
    // for local_decl in &body.local_decls {
    // println!("local decl: {:#?}", body.local_decls);
    println!("arg_count: {:#?}", body.arg_count);

    for arg_local in body.args_iter() {
        // function arguments
        let decl = &body.local_decls[arg_local];
        // 判断是否可mutate
        if decl.ty.is_mutable_ptr() && !decl.ty.is_primitive_ty() {
            println!("arg: {:#?}", arg_local);
            // println!("decl: {:#?}", decl);
            let mutate_operation_id = StatementId::new(fn_id, arg_local);
            let mutate_operation_info = StatementInfo {
                type_name: (OperationType::Mutate, body.local_decls[arg_local].ty.to_string()),
                src: None,
                span: decl.source_info.span,
                gen_bbs: Vec::new(),
                kill_bbs: Vec::new(),
            };
            drop_operations.insert(mutate_operation_id, mutate_operation_info);
        }
    }

    let mut def_use_analysis = DefUseAnalysis::new(body);
    def_use_analysis.analyze(body);
    let drop_operations = collect_operation_src_info(drop_operations, body, &def_use_analysis);
    collect_gen_kill_bbs(drop_operations, body, &def_use_analysis)
}

pub fn collect_input_params_info(
    fn_id: LocalDefId,
    body: &Body<'_>,
    input_num: usize,
) -> Vec<String> {
    let mut input_types: Vec<String> = Vec::new();
    if input_num <= 0 { return input_types;}
    for i in 0..input_num {
        let local = Local::from_usize(i+1);
        let input_type = body.local_decls[local].ty.to_string();
        input_types.push(input_type);
        // println!("input_type: {}", input_type);
    }
    input_types
}

pub fn collect_output_param_info(
    fn_id: LocalDefId,
    body: &Body<'_>,
) -> Option<String> {
    let local = Local::from_usize(body.local_decls.len() - 1);
    let output_type = body.local_decls[local].ty.to_string();
    let output_type = facilitate_type_name(&output_type);
    Some(output_type)
}

pub fn collect_func_params_info(
    fn_id: LocalDefId,
    body: &Body<'_>,
) -> HashSet<String> {
    let mut param_types: HashSet<String> = HashSet::new();
    for local_decl in &body.local_decls { 
        let mut struct_type = local_decl.ty.to_string();
        struct_type = facilitate_type_name(&struct_type);
        param_types.insert(struct_type);
    }
    param_types
}

pub fn collect_return_info(
    fn_id: LocalDefId,
    body: &Body<'_>,
) -> HashMap<StatementId, StatementInfo> {

    let mut return_map: HashMap<StatementId, StatementInfo> = HashMap::new();

    println!("Collect return Info:");
    let local_decls = &body.local_decls;
    
    let return_decl = &local_decls[Local::from_usize(0)];
    // if return_ty {
    println!("return ty: {:?}", return_decl.ty);
    println!("return ty kind: {:?}", return_decl.ty.kind());
    if return_decl.ty.to_string() != String::from("()") {
        let return_id = StatementId::new(fn_id, Local::from_usize(0));
        let return_info = StatementInfo {
            type_name: (OperationType::Return, return_decl.ty.to_string()),
            src: None,
            span: return_decl.source_info.span,
            gen_bbs: Vec::new(),
            kill_bbs: Vec::new(),
        };
        return_map.insert(return_id, return_info);
    }

    let mut def_use_analysis = DefUseAnalysis::new(body);
    def_use_analysis.analyze(body);

    let return_map = collect_operation_src_info(return_map, body, &def_use_analysis);
    collect_gen_kill_bbs(return_map, body, &def_use_analysis)
}

fn collect_gen_kill_bbs(
    statememts: HashMap<StatementId, StatementInfo>,
    _body: &Body<'_>,
    def_use_analysis: &DefUseAnalysis,
) -> HashMap<StatementId, StatementInfo> {
    if statememts.is_empty() {
        return statememts;
    }
    statememts
        .into_iter()
        .filter_map(|(id, mut info)| {
            let use_info = def_use_analysis.local_info(id.local);
            for u in &use_info.defs_and_uses {
                match u.context {
                    PlaceContext::NonUse(context) => match context {
                        NonUseContext::StorageLive => { info.gen_bbs.push(u.location.block); },
                        NonUseContext::StorageDead => { info.kill_bbs.push(u.location.block); },
                        _ => {}
                    },
                    PlaceContext::NonMutatingUse(context) => {
                        if let NonMutatingUseContext::Move = context {
                            info.kill_bbs.push(u.location.block);
                        }
                    }
                    PlaceContext::MutatingUse(context) => match context {
                        MutatingUseContext::Drop => info.kill_bbs.push(u.location.block),
                        MutatingUseContext::Store => {}
                        MutatingUseContext::Call => {}
                        _ => {}
                    },
                }
            }
            Some((id, info))
        })
        .collect::<HashMap<_, _>>()
}

fn collect_operation_src_info(
    statememts: HashMap<StatementId, StatementInfo>,
    body: &Body<'_>,
    def_use_analysis: &DefUseAnalysis,
) -> HashMap<StatementId, StatementInfo> {
    if statememts.is_empty() {
        return statememts;
    }
    let batch_depends = batch_gen_depends_for_all(&statememts, body, def_use_analysis);
    // println!("batch depends: {:#?}", batch_depends);
    statememts
        .into_iter()
        .map(|(id, mut info)| {
            let (place, tracker_result) = match info.type_name.0 {
                OperationType::Drop 
                | OperationType::GetRawPtr 
                | OperationType::UnsafeBlock
                | OperationType::Mutate
                | OperationType::Return => {
                    let mut tracker = Tracker::new(Place::from(id.local), true, &batch_depends);
                    tracker.track()
                }
                OperationType::StdMutexGuard | OperationType::StdRwLockGuard => {
                    let mut tracker = Tracker::new(Place::from(id.local), false, &batch_depends);
                    tracker.track()
                }
            };
            println!("place: {:?}, tracker result: {:?}", place, tracker_result);
            info.src = match tracker_result {
                TrackerState::Init => {
                    let fields = place
                        .projection
                        .iter()
                        .filter_map(|e| {
                            if let ProjectionElem::Field(field, _) = e {
                                Some(field)
                            } else {
                                None
                            }
                        })
                        .fold(String::new(), |acc, field| {
                            acc + &format!("{:?}", field) + ","
                        });
                    let struct_type = body.local_decls[place.local].ty.to_string();
                    let simplified_type = facilitate_type_name(&struct_type);
                    let param_src = StatememtSrc::Init(InitContext {
                        struct_type,
                        simplified_type,
                        fields,
                        local: place.local,
                    });
                    Some(param_src)
                }
                TrackerState::ParamSrc => {
                    let fields = place
                        .projection
                        .iter()
                        .filter_map(|e| {
                            if let ProjectionElem::Field(field, _) = e {
                                Some(field)
                            } else {
                                None
                            }
                        })
                        .fold(String::new(), |acc, field| {
                            acc + &format!("{:?}", field) + ","
                        });
                    let struct_type = body.local_decls[place.local].ty.to_string();
                    let simplified_type = facilitate_type_name(&struct_type);
                    let param_src = StatememtSrc::ParamSrc(ParamSrcContext {
                        struct_type,
                        simplified_type,
                        fields,
                        local: place.local,
                    });
                    Some(param_src)
                }
                TrackerState::LocalSrc => {
                    let param_src = StatememtSrc::LocalSrc(LocalSrcContext {
                        place: format!("{:?}", place),
                        local: place.local,
                    });
                    Some(param_src)
                }
                TrackerState::WrapperLock => {
                    match body.local_decls[place.local].local_info {
                        Some(box LocalInfo::StaticRef {
                            def_id,
                            is_thread_local: _,
                        }) => {
                            let param_src = StatememtSrc::GlobalSrc(GlobalSrcContext { 
                                global_id: def_id,
                                local: place.local,
                            });
                            Some(param_src)
                        }
                        _ => {
                            // TODO(boqin): any other non-static-ref lock wrapper?
                            None
                        }
                    }
                }
                _ => None,
            };
            (id, info)
        })
        .collect()
}

fn batch_gen_depends_for_all<'a, 'b, 'tcx>(
    statememts: &HashMap<StatementId, StatementInfo>,
    body: &'a Body<'tcx>,
    def_use_analysis: &'b DefUseAnalysis,
) -> BatchDependResults<'a, 'b, 'tcx> {
    let mut batch_depend_results = BatchDependResults::new(body, def_use_analysis);
    for id in statememts.keys() {
        batch_gen_depends(id.local, &mut batch_depend_results);
    }
    batch_depend_results
}

fn batch_gen_depends(local: Local, batch_depend_results: &mut BatchDependResults<'_, '_, '_>) {
    let local_place = Place::from(local);
    let mut worklist: Vec<Place<'_>> = vec![local_place];
    let mut visited: HashSet<Local> = HashSet::new();
    visited.insert(local);
    while let Some(place) = worklist.pop() {
        batch_depend_results.gen_depends(place);
        for depend in batch_depend_results
            .get_depends(place)
            .into_iter()
            .map(|(place, _)| place)
        {
            if !visited.contains(&depend.local) {
                worklist.push(depend);
                visited.insert(depend.local);
            }
        }
    }
    // println!("visited local: {:#?}", visited);
}