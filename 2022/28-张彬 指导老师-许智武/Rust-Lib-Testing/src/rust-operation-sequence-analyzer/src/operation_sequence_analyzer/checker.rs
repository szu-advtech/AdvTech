extern crate rustc_hir;
extern crate rustc_middle;
extern crate rustc_span;
use super::callgraph::Callgraph;
use super::collector::{collect_rawptr_info, collect_unsafeconstruct_info, collect_drop_operation_info};
use super::config::CrateNameLists;
//use super::config::CALLCHAIN_DEPTH;
//use super::genkill::GenKill;
use super::operationsequence::OperationSequenceInfo;
use super::operationsequence::{StatementId, StatementInfo};
use rustc_hir::def_id::LocalDefId;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_hir::FnDecl; // Represents the header (not the body) of a function declaration.
use rustc_middle::mir::{Safety};
use rustc_middle::ty::TyCtxt;
// use rustc_middle::hir::map::Map;
use rustc_span::Span;

use std::cell::RefCell;
use std::collections::HashMap;
//use std::collections::HashSet;
pub struct OperationSequenceAnalyzer {
    crate_name_lists: CrateNameLists,
    crate_operationsequences: HashMap<StatementId, StatementInfo>,
    crate_callgraph: Callgraph,
    crate_lock_pairs: RefCell<Vec<OperationSequenceInfo>>,
}

impl OperationSequenceAnalyzer {
    pub fn new(is_white: bool, crate_name_lists: Vec<String>) -> Self {
        println!("[ROSA] Creating OperationSequenceAnalyzer.");
        if is_white {
            println!("[ROSA] Is_white.");
            Self {
                crate_name_lists: CrateNameLists::White(crate_name_lists),
                crate_operationsequences: HashMap::new(),
                crate_callgraph: Callgraph::new(),
                crate_lock_pairs: RefCell::new(Vec::new()),
            }
        } else {
            println!("[ROSA] Is_black.");
            Self {
                crate_name_lists: CrateNameLists::Black(crate_name_lists),
                crate_operationsequences: HashMap::new(),
                crate_callgraph: Callgraph::new(),
                crate_lock_pairs: RefCell::new(Vec::new()),
            }
        }
    }
    pub fn check(&mut self, tcx: TyCtxt) {
        let crate_name = tcx.crate_name(LOCAL_CRATE).to_string();
        match &self.crate_name_lists {
            CrateNameLists::White(lists) => {
                println!("[ROSA] Crate {} is in white name list.",crate_name);
                if !lists.contains(&crate_name) {
                    return;
                }
            }
            CrateNameLists::Black(lists) => {
                println!("[ROSA] Crate {} is in black name list.",crate_name);
                if lists.contains(&crate_name) {
                    return;
                }
            }
        }

        // collect fn
        let ids = tcx.mir_keys(LOCAL_CRATE);
        let fn_ids: Vec<LocalDefId> = ids
            .clone()
            .into_iter()
            .filter(|id| {
                let hir = tcx.hir();
                hir.body_owner_kind(hir.local_def_id_to_hir_id(*id))
                    .is_fn_or_closure()
            })
            .collect();
        // println!("[ROSA] fn_ids: {:#?}", fn_ids);
        
        // generate callgraph
        let hir = tcx.hir();
        for fn_id in &fn_ids {
            let fn_decl: &FnDecl = hir.fn_decl_by_hir_id(hir.local_def_id_to_hir_id(*fn_id)).unwrap();
            // println!("Fn declaration: {:?}", fn_decl);
            self.crate_callgraph
                .generate(*fn_id, tcx.optimized_mir(*fn_id), &fn_ids);
        }

        // print callgraph
        println!("---------------------------------------");
        println!("[ROSA] Printing callgraph:");
        self.crate_callgraph._print();

        // collect information of unsafe construct
        let unsafeconstructs: HashMap<LocalDefId, HashMap<Span, Safety>> = fn_ids
            .clone()
            .into_iter()
            .filter_map(|fn_id| {
                let body = tcx.optimized_mir(fn_id);
                let unsafeconstructs = collect_unsafeconstruct_info(body);
                if unsafeconstructs.is_empty() {
                    None
                } else {
                    Some((fn_id, unsafeconstructs))
                }
            })
            .collect();
        
        if unsafeconstructs.is_empty() {
            return;
        }
        println!("---------------------------------------");
        println!("[ROSA] unsafeconstructs: {:#?}", unsafeconstructs);

        // collect information of rawptr
        let rawptrs: HashMap<LocalDefId, HashMap<StatementId, StatementInfo>> = fn_ids
            .clone()
            .into_iter()
            .filter_map(|fn_id| {
                let body = tcx.optimized_mir(fn_id);
                let rawptrs = collect_rawptr_info(fn_id, body);
                if rawptrs.is_empty() {
                    None
                } else {
                    Some((fn_id, rawptrs))
                }
            })
            .collect();
        
        if rawptrs.is_empty() {
            return;
        }

        // print information of rawptr
        println!("---------------------------------------");
        for (_, info) in rawptrs.iter() {
            self.crate_operationsequences.extend(info.clone().into_iter());
        }
        println!(
            "[ROSA] fn with rawptrs: {}, rawptrs num: {}, local fn num: {}",
            rawptrs.len(),
            self.crate_operationsequences.len(),
            fn_ids.len()
        );
        println!("[ROSA] rawptrs: {:#?}", rawptrs);

        // collect information of drop operation
        let drop_operations: HashMap<LocalDefId, HashMap<StatementId, StatementInfo>> = fn_ids
        .clone()
        .into_iter()
        .filter_map(|fn_id| {
            let body = tcx.optimized_mir(fn_id);
            let drop_operations = collect_drop_operation_info(fn_id, body);
            if drop_operations.is_empty() {
                None
            } else {
                Some((fn_id, drop_operations))
            }
        })
        .collect();
    
        if drop_operations.is_empty() {
            return;
        }

        // print information of drop operation
        println!("---------------------------------------");
        for (_, info) in drop_operations.iter() {
            self.crate_operationsequences.extend(info.clone().into_iter());
        }
        println!(
            "[ROSA] fn with drop_operations: {}, drop_operations num: {}, local fn num: {}",
            drop_operations.len(),
            self.crate_operationsequences.len(),
            fn_ids.len()
        );
        println!("[ROSA] drop_operations: {:#?}", drop_operations);

        println!("[ROSA] Finish OperationSequenceAnalyzer.check().");
    }

}
