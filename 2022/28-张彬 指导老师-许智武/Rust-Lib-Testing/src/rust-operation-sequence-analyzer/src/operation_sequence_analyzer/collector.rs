extern crate rustc_hir;
extern crate rustc_middle;
extern crate rustc_mir;
extern crate rustc_span;

use super::dataflow::*;
use super::def_use::DefUseAnalysis;
use super::operationsequence::*;
use super::operationsequence::{OperationSequenceInfo, StatementId, StatementInfo};
use super::tracker::{Tracker, TrackerState};
use rustc_hir::def_id::LocalDefId;
use rustc_middle::mir::visit::{
    MutatingUseContext, NonMutatingUseContext, NonUseContext, PlaceContext,
};
use rustc_middle::mir::{StatementKind, Body, Local, LocalInfo, Place, ProjectionElem, Rvalue, ClearCrossCrate, Safety};
use rustc_middle::mir::terminator::TerminatorKind;
use std::collections::{HashMap, HashSet};
use rustc_span::Span;

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

pub fn collect_unsafeconstruct_info(body: &Body) -> HashMap<Span, Safety> {
    let mut unsafeconstruct: HashMap<Span, Safety> = HashMap::new();
    // find unsafe construct
    for each_source_scope_data in &body.source_scopes {
        if let ClearCrossCrate::Set(source_scope_local_data) = &each_source_scope_data.local_data  {
            if let Safety::Safe = source_scope_local_data.safety {} else {
                //println!("{:?}", each_source_scope_data);
                //println!("{:?}, {:?}", &each_source_scope_data.span, &each_source_scope_data.local_data);
                unsafeconstruct.insert(each_source_scope_data.span, source_scope_local_data.safety);
            }            
        }
    }
    unsafeconstruct
}

pub fn collect_rawptr_info(
    fn_id: LocalDefId,
    body: &Body,
) -> HashMap<StatementId, StatementInfo> {
    let mut rawptrs: HashMap<StatementId, StatementInfo> = HashMap::new();
    
    // find raw pointer
    for each_basicblock in body.basic_blocks() {
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
        }
    }

    let mut def_use_analysis = DefUseAnalysis::new(body);
    def_use_analysis.analyze(body);
    let rawptrs = collect_operation_src_info(rawptrs, body, &def_use_analysis);
    collect_gen_kill_bbs(rawptrs, body, &def_use_analysis)
}

pub fn collect_drop_operation_info(
    fn_id: LocalDefId,
    body: &Body,
) -> HashMap<StatementId, StatementInfo> {
    let mut drop_operations: HashMap<StatementId, StatementInfo> = HashMap::new();
    
    // find drop_operation
    for each_basicblock in body.basic_blocks() {
        if let Some(terminator) = &each_basicblock.terminator {
            if let TerminatorKind::Drop {place: tkd_place, target: tkd_target, unwind: tkd_unwind,} = &terminator.kind {               
                if let Some(last_statement_in_bb) = &each_basicblock.statements.last() {
                    if let StatementKind::StorageDead(storagedeallocal) = last_statement_in_bb.kind {
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
            }
        }
    }

    let mut def_use_analysis = DefUseAnalysis::new(body);
    def_use_analysis.analyze(body);
    let drop_operations = collect_operation_src_info(drop_operations, body, &def_use_analysis);
    collect_gen_kill_bbs(drop_operations, body, &def_use_analysis)
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

fn batch_gen_depends(local: Local, batch_depend_results: &mut BatchDependResults) {
    let local_place = Place::from(local);
    let mut worklist: Vec<Place> = vec![local_place];
    let mut visited: HashSet<Place> = HashSet::new();
    visited.insert(local_place);
    while let Some(place) = worklist.pop() {
        batch_depend_results.gen_depends(place);
        for depend in batch_depend_results
            .get_depends(place)
            .into_iter()
            .map(|(place, _)| place)
        {
            if !visited.contains(&depend) {
                worklist.push(depend);
                visited.insert(depend);
            }
        }
    }
}

fn collect_operation_src_info(
    statememts: HashMap<StatementId, StatementInfo>,
    body: &Body,
    def_use_analysis: &DefUseAnalysis,
) -> HashMap<StatementId, StatementInfo> {
    if statememts.is_empty() {
        return statememts;
    }
    let batch_depends = batch_gen_depends_for_all(&statememts, body, def_use_analysis);
    // println!("Batch Dependes: {:#?}", batch_depends);
    statememts
        .into_iter()
        .map(|(id, mut info)| {
            // println!("Type Name: {:?}", info.type_name.0);
            let (place, tracker_result) = match info.type_name.0 {
                OperationType::GetRawPtr | OperationType::StdMutexGuard | OperationType::StdRwLockGuard => {
                    let mut tracker = Tracker::new(Place::from(id.local), true, &batch_depends);
                    tracker.track()
                }
                _ => {
                    let mut tracker = Tracker::new(Place::from(id.local), false, &batch_depends);
                    tracker.track()
                }
            };
            // println!("After Track: {:?}->{:?},\n {:?}", id.local, place, tracker_result);
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
                    let mut struct_type = body.local_decls[place.local].ty.to_string();
                    if struct_type.starts_with('&') {
                        struct_type = struct_type.chars().skip(1).collect();
                    }
                    let lockguard_src = StatememtSrc::Init(InitContext {
                        struct_type,
                        fields,
                    });
                    Some(lockguard_src)
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
                    let mut struct_type = body.local_decls[place.local].ty.to_string();
                    if struct_type.starts_with('&') {
                        struct_type = struct_type.chars().skip(1).collect();
                    }
                    let lockguard_src = StatememtSrc::ParamSrc(ParamSrcContext {
                        struct_type,
                        fields,
                    });
                    Some(lockguard_src)
                }
                TrackerState::LocalSrc => {
                    let lockguard_src = StatememtSrc::LocalSrc(LocalSrcContext {
                        place: format!("{:?}", place),
                    });
                    Some(lockguard_src)
                }
                TrackerState::WrapperLock => {
                    match body.local_decls[place.local].local_info {
                        Some(box LocalInfo::StaticRef {
                            def_id,
                            is_thread_local: _,
                        }) => {
                            let lockguard_src = StatememtSrc::GlobalSrc(GlobalSrcContext { global_id: def_id });
                            Some(lockguard_src)
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

fn collect_gen_kill_bbs(
    statememts: HashMap<StatementId, StatementInfo>,
    _body: &Body,
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
