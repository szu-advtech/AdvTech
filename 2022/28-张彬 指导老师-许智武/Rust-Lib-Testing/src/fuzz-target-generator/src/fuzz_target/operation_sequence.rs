use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::mir::{BasicBlock, Local, Mutability};
use rustc_middle::ty::Ty;
use rustc_span::Span;

use std::hash::Hash;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum StatememtSrc {
    Init(InitContext),
    ParamSrc(ParamSrcContext),
    LocalSrc(LocalSrcContext),
    GlobalSrc(GlobalSrcContext),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct InitContext {
    pub struct_type: String,
    pub simplified_type: String,
    pub fields: String,
    pub local: Local,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ParamSrcContext {
    pub struct_type: String,
    pub simplified_type: String,
    pub fields: String,
    pub local: Local,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct LocalSrcContext {
    pub place: String,
    pub local: Local,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct GlobalSrcContext {
    pub global_id: DefId,
    pub local: Local,
}

#[derive(Debug, Clone)]
pub struct StatementInfo {
    pub type_name: (OperationType, String),
    pub src: Option<StatememtSrc>,
    pub span: Span,
    pub gen_bbs: Vec<BasicBlock>,
    pub kill_bbs: Vec<BasicBlock>,
}

impl PartialEq for StatementInfo {
    fn eq(&self, other: &Self) -> bool {
        self.type_name == other.type_name
            && if let Some(self_src) = &self.src {
                if let Some(other_src) = &other.src {
                    *self_src == *other_src
                } else {
                    false
                }
            } else {
                false
            }
    }
}
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub struct StatementId {
    pub fn_id: LocalDefId,
    pub local: Local,
}

impl StatementId {
    pub fn new(fn_id: LocalDefId, local: Local) -> Self {
        Self { fn_id, local }
    }
}
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum OperationType {
    GetRawPtr,
    Drop,
    StdMutexGuard,
    StdRwLockGuard,
    UnsafeBlock,
    Mutate,
    Return,
}

pub fn parse_lockguard_type(ty: &Ty<'_>) -> Option<(OperationType, String)> {
    let type_name = ty.to_string();
    if type_name.starts_with("std::sync::MutexGuard<") {
        Some((
            OperationType::StdMutexGuard,
            extract_data_type("std::sync::MutexGuard<", &type_name),
        ))
    } else if type_name.starts_with("sync::mutex::MutexGuard<") {
        Some((
            OperationType::StdMutexGuard,
            extract_data_type("sync::mutex::MutexGuard<", &type_name),
        ))
    } else if type_name.starts_with("std::sync::RwLockReadGuard<") {
        Some((
            OperationType::StdRwLockGuard,
            extract_data_type("std::sync::RwLockReadGuard<", &type_name),
        ))
    } else if type_name.starts_with("std::sync::RwLockWriteGuard<") {
        Some((
            OperationType::StdRwLockGuard,
            extract_data_type("std::sync::RwLockWriteGuard<", &type_name),
        ))
    } else {
        None
    }
}

fn extract_data_type(lockguard_type: &str, type_name: &str) -> String {
    assert!(type_name.starts_with(lockguard_type) && type_name.ends_with('>'));
    type_name[lockguard_type.len()..type_name.len() - 1].to_string()
}

#[test]
fn test_extract_data_type() {
    assert!(
        extract_data_type(
            "std::sync::MutexGuard<",
            "std::sync::MutexGuard<std::vec::Vec<Foo>>"
        ) == "std::vec::Vec<Foo>"
    );
    assert!(
        extract_data_type(
            "lock_api::mutex::MutexGuard<",
            "lock_api::mutex::MutexGuard<parking_lot::raw_mutex::RawMutex, i32>"
        ) == "parking_lot::raw_mutex::RawMutex, i32"
    );
}
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct OperationSequenceInfo {
    pub first: StatementId,
    pub second: StatementId,
}
