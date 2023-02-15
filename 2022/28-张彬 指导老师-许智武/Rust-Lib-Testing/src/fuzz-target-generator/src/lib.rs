#![feature(rustc_private)]
#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(in_band_lifetimes)]
#![feature(nll)]
#![feature(test)]
#![feature(crate_visibility_modifier)]
#![feature(never_type)]
#![feature(int_log)]
#![feature(mutex_unlock)]
#![recursion_limit = "256"]
#![allow(dead_code, unused_imports, unused_variables)]

extern crate rustc_hir;
extern crate rustc_middle;
extern crate rustc_span;
extern crate rustc_index;
extern crate rustc_ast;
extern crate rustc_data_structures;

use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_hir::HirId;
use std::collections::HashMap;

pub mod fuzz_target {
    pub mod afl_util;
    pub mod api_function;
    pub mod api_graph;
    pub mod api_sequence;
    pub mod api_parameter;
    pub mod api_util;
    pub mod call_type;
    pub mod file_util;
    pub mod fuzzable_type;
    pub mod generic_function;
    pub mod impl_util;
    pub mod mod_visibility;
    pub mod prelude_type;
    pub mod print_message;
    pub mod replay_util;
    pub mod operation_sequence;
    pub mod collector;
    pub mod def_use;
    pub mod tracker;
    pub mod dataflow;
}

pub fn call_rust_doc_main() {
    rustdoc::main();
}

pub struct ApiDependencyVisitor {
    items: HashMap<String, HirId>,
}

impl ApiDependencyVisitor {
    pub fn new() -> ApiDependencyVisitor {
        ApiDependencyVisitor {
            items: HashMap::new(),
        }
    }
}

impl<'hir> ItemLikeVisitor<'hir> for ApiDependencyVisitor {
    fn visit_item(&mut self, item: &'hir rustc_hir::Item<'hir>) {
        self.items.insert(item.ident.to_string(), item.hir_id());
        //println!("visit: {}", item.ident.to_string());
    }

    fn visit_trait_item(&mut self, _trait_item: &'hir rustc_hir::TraitItem<'hir>) {}

    fn visit_impl_item(&mut self, _impl_item: &'hir rustc_hir::ImplItem<'hir>) {}

    fn visit_foreign_item(&mut self, _impl_item: &'hir rustc_hir::ForeignItem<'hir>) {}
}
