/// A simple assumption to track lockguard to lock:
/// For two places: A and B
/// if A = move B:
/// then A depends on B by move
/// if A = B:
/// then A depends on B by copy
/// if A = &B or A = &mut B
/// then A depends on B by ref
/// if A = call func(move B)
/// then A depends on B by call

use crate::fuzz_target::def_use::DefUseAnalysis;
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;

use std::collections::HashMap;

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct DependPair<'tcx>(Place<'tcx>, Place<'tcx>);

pub type DependCache<'tcx> = HashMap<DependPair<'tcx>, DependResult>;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum DependResult {
    MoveDepend,
    CopyDepend,
    RefDepend,
    CallDepend,
    AddressOfDepend,
    LenDepend,
    DiscriminantDepend,
}

pub struct BatchDependResults<'a, 'b, 'tcx> {
    depend_query_info: DependQueryInfo<'tcx>,
    pub body: &'a Body<'tcx>,
    def_use_analysis: &'b DefUseAnalysis,
}

impl<'a, 'b, 'tcx> BatchDependResults<'a, 'b, 'tcx> {
    pub fn new(body: &'a Body<'tcx>, def_use_analysis: &'b DefUseAnalysis) -> Self {
        Self {
            depend_query_info: DependQueryInfo::<'tcx>::new(),
            body,
            def_use_analysis,
        }
    }

    pub fn get_depends(&self, place: Place<'tcx>) -> Vec<(Place<'tcx>, DependResult)> {
        self.depend_query_info.get_depends(place)
    }

    pub fn gen_depends(&mut self, place: Place<'tcx>) {
        // println!("gen depends place: {:#?}", place);
        let use_info = self.def_use_analysis.local_info(place.local);
        for u in &use_info.defs_and_uses {
            // println!("context: {:#?}", u.context);
            match u.context {
                PlaceContext::MutatingUse(MutatingUseContext::Store | MutatingUseContext::Projection) => {
                    if is_terminator_location(&u.location, self.body) {
                        continue;
                    }
                    let stmt = &self.body.basic_blocks()[u.location.block].statements
                        [u.location.statement_index];
                    // println!("stmt: {:#?}\nstmt kind: {:#?}", stmt, stmt.kind);
                    if let StatementKind::Assign(box (lhs, ref rvalue)) = stmt.kind {
                        // println!("rvalue: {:#?}", rvalue);
                        if lhs.local != place.local {
                            println!("lhs: {:?}, place: {:?}", lhs, place);
                            continue;
                        }
                        match rvalue {
                            Rvalue::Use(operand) => {
                                match operand {
                                    Operand::Move(rhs) => {
                                        self.depend_query_info.add_depend(
                                            DependPair(lhs, *rhs),
                                            DependResult::MoveDepend,
                                        );
                                    }
                                    Operand::Copy(rhs) => {
                                        self.depend_query_info.add_depend(
                                            DependPair(lhs, *rhs),
                                            DependResult::CopyDepend,
                                        );
                                    }
                                    _ => {
                                        // TODO
                                    }
                                };
                            }
                            Rvalue::Ref(_, _, rhs) => {
                                self.depend_query_info
                                    .add_depend(DependPair(lhs, Place::from(rhs.local)), DependResult::RefDepend);
                            }
                            Rvalue::Repeat(operand, _) => {
                                match operand {
                                    Operand::Move(rhs) => {
                                        self.depend_query_info.add_depend(
                                            DependPair(lhs, *rhs),
                                            DependResult::MoveDepend,
                                        );
                                    }
                                    Operand::Copy(rhs) => {
                                        self.depend_query_info.add_depend(
                                            DependPair(lhs, *rhs),
                                            DependResult::CopyDepend,
                                        );
                                    }
                                    _ => {
                                        // TODO
                                    }
                                };
                            }
                            Rvalue::AddressOf(_, rhs) => {
                                println!("rvalue addressOf");
                                self.depend_query_info.add_depend(
                                    DependPair(lhs, *rhs),
                                    DependResult::AddressOfDepend,
                                );
                            }
                            Rvalue::Len(rhs) => {
                                self.depend_query_info.add_depend(
                                    DependPair(lhs, *rhs),
                                    DependResult::LenDepend,
                                );
                            }
                            Rvalue::Discriminant(rhs) => {
                                self.depend_query_info.add_depend(
                                    DependPair(lhs, *rhs),
                                    DependResult::DiscriminantDepend,
                                );
                            }
                            Rvalue::ShallowInitBox(operand, _) => {
                                match operand {
                                    Operand::Move(rhs) => {
                                        self.depend_query_info.add_depend(
                                            DependPair(lhs, *rhs),
                                            DependResult::MoveDepend,
                                        );
                                    }
                                    Operand::Copy(rhs) => {
                                        self.depend_query_info.add_depend(
                                            DependPair(lhs, *rhs),
                                            DependResult::CopyDepend,
                                        );
                                    }
                                    _ => {
                                        // TODO
                                    }
                                };
                            }
                            Rvalue::Cast(_, operand, _) => {
                                match operand {
                                    Operand::Move(rhs) => {
                                        self.depend_query_info.add_depend(
                                            DependPair(lhs, *rhs),
                                            DependResult::MoveDepend,
                                        );
                                    }
                                    Operand::Copy(rhs) => {
                                        self.depend_query_info.add_depend(
                                            DependPair(lhs, *rhs),
                                            DependResult::CopyDepend,
                                        );
                                    }
                                    _ => {
                                        // TODO
                                    }
                                };
                            }
                            Rvalue::UnaryOp(_, operand) => {
                                match operand {
                                    Operand::Move(rhs) => {
                                        self.depend_query_info.add_depend(
                                            DependPair(lhs, *rhs),
                                            DependResult::MoveDepend,
                                        );
                                    }
                                    Operand::Copy(rhs) => {
                                        self.depend_query_info.add_depend(
                                            DependPair(lhs, *rhs),
                                            DependResult::CopyDepend,
                                        );
                                    }
                                    _ => {
                                        // TODO
                                    }
                                };
                            }
                            _ => {
                                // TODO
                                println!("Other Rvalue Type: \n{:#?}", rvalue);
                            }
                        }
                    }
                }
                PlaceContext::MutatingUse(MutatingUseContext::Call) => {
                    assert!(is_terminator_location(&u.location, self.body));
                    let term = self.body.basic_blocks()[u.location.block].terminator();
                    // println!("term: {:#?}\nterm kind: {:#?}", term, term.kind);
                    if let TerminatorKind::Call {
                        func: _,
                        ref args,
                        destination: Some((lhs, _)),
                        ..
                    } = term.kind
                    {
                        if lhs.local != place.local {
                            // println!("lhs: {:?}, place: {:?}", lhs, place);
                            continue;
                        }
                        // heuristically consider the first move arg to be associated with return.
                        // TODO: check the type relations to decide if they are related.
                        for arg in args {
                            if let Operand::Move(rhs) = arg {
                                self.depend_query_info
                                    .add_depend(DependPair(lhs, *rhs), DependResult::CallDepend);
                                break;
                            }
                        }
                    }
                }
                // PlaceContext::MutatingUse(MutatingUseContext::Projection) => {
                //     println!("Projection placeContext: {:?}, place: {:?}, location: {:?}", u.context, place, u.location);
                //     assert!(!is_terminator_location(&u.location, self.body));
                //     let stmt = &self.body.basic_blocks()[u.location.block].statements
                //         [u.location.statement_index];
                //     let term = self.body.basic_blocks()[u.location.block].terminator();
                //     println!("projection stmt: {:#?}", stmt);
                //     println!("projection term: {:#?}", term);
                // }
                PlaceContext::NonMutatingUse(NonMutatingUseContext::Move) => {
                    println!("Move placeContext: {:?}, place: {:?}, location: {:?}", u.context, place, u.location);
                    // println!("Location: {:#?}", u.location);
                    // assert!(!is_terminator_location(&u.location, self.body));
                    // let stmt = &self.body.basic_blocks()[u.location.block].statements
                    //     [u.location.statement_index];
                    // println!("stmt: {:#?}\nstmt kind: {:#?}", stmt, stmt.kind);
                }
                _ => { println!("Other placeContext: {:?}, place: {:?}, location: {:?}", u.context, place, u.location); }
            }
        }
    }
}

pub struct DependQueryInfo<'tcx> {
    depend_cache: DependCache<'tcx>,
}

impl<'tcx> DependQueryInfo<'tcx> {
    pub fn new() -> Self {
        Self {
            depend_cache: HashMap::<DependPair<'tcx>, DependResult>::new(),
        }
    }

    pub fn get_depends(&self, place: Place<'tcx>) -> Vec<(Place<'tcx>, DependResult)> {
        // println!("depend cache: {:#?}", self.depend_cache);
        self.depend_cache
            .iter()
            .filter_map(|(pair, result)| {
                // println!("pair.0:{:#?}\npair.1:{:#?}", pair.0, pair.1);
                if pair.0.local == place.local {
                    Some((pair.1, *result))
                } else {
                    None
                }
            })
            .collect()
    }

    fn add_depend(&mut self, pair: DependPair<'tcx>, result: DependResult) {
        self.depend_cache.entry(pair).or_insert(result);
    }
}

fn is_terminator_location(location: &Location, body: &Body<'_>) -> bool {
    location.statement_index >= body.basic_blocks()[location.block].statements.len()
}
