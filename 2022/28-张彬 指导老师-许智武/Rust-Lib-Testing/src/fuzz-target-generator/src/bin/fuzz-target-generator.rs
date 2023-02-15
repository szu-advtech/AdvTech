#![feature(rustc_private)]
#![feature(box_patterns)]
#![feature(once_cell)]
#![allow(dead_code, unused_imports, unused_variables)]

extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_interface;
extern crate rustc_lint;
extern crate rustc_middle;
extern crate rustc_passes;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustdoc;
#[macro_use]
extern crate log;

use fuzz_target_generator::ApiDependencyVisitor;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_driver::{abort_on_err, describe_lints};
use rustc_errors::ErrorReported;
use rustc_interface::interface;
use rustc_middle::middle::privacy::AccessLevels;
use rustc_middle::ty::{ParamEnv, Ty, TyCtxt};
use rustc_session::config::{self, make_crate_type_option, ErrorOutputType, RustcOptGroup};
use rustc_session::getopts;
use rustc_session::{early_error, early_warn};
use rustc_span::symbol::sym;
use rustc_span::Span;
use rustc_span::Symbol;
use rustdoc::clean::{self, TraitWithExtraInfo};
use rustdoc::config::{self as rdConfig, OutputFormat, RenderOptions};
use rustdoc::core::{self, DocContext};
use rustdoc::doctest::{self, Collector};
use rustdoc::error::Error;
use rustdoc::formats;
use rustdoc::formats::renderer;
use rustdoc::formats::{cache::Cache, item_type::ItemType};
use rustdoc::html::{self, markdown, render::Context};
use rustdoc::json;
use rustdoc::passes::{self, Condition::*, ConditionalPass};
use rustdoc::scrape_examples;
use std::{process, env};
use std::time::{Instant, SystemTime};
use std::collections::{HashMap, HashSet};

// extern crate fuzz_target;
use fuzz_target_generator::fuzz_target::{api_function, api_graph, api_util, file_util, impl_util, print_message};

use std::cell::RefCell;
use std::lazy::SyncLazy;
use std::mem;
use std::rc::Rc;

fn main() {
    let start = Instant::now();
    println!("Fuzz Target Generator for Rust Libraries: v0.2.0");

    // about jemalloc-sys
    #[cfg(feature = "jemalloc")]
    {
        use std::os::raw::{c_int, c_void};

        #[used]
        static _F1: unsafe extern "C" fn(usize, usize) -> *mut c_void = jemalloc_sys::calloc;
        #[used]
        static _F2: unsafe extern "C" fn(*mut *mut c_void, usize, usize) -> c_int =
            jemalloc_sys::posix_memalign;
        #[used]
        static _F3: unsafe extern "C" fn(usize, usize) -> *mut c_void = jemalloc_sys::aligned_alloc;
        #[used]
        static _F4: unsafe extern "C" fn(usize) -> *mut c_void = jemalloc_sys::malloc;
        #[used]
        static _F5: unsafe extern "C" fn(*mut c_void, usize) -> *mut c_void = jemalloc_sys::realloc;
        #[used]
        static _F6: unsafe extern "C" fn(*mut c_void) = jemalloc_sys::free;

        // On OSX, jemalloc doesn't directly override malloc/free, but instead
        // registers itself with the allocator's zone APIs in a ctor. However,
        // the linker doesn't seem to consider ctors as "used" when statically
        // linking, so we need to explicitly depend on the function.
        #[cfg(target_os = "macos")]
        {
            extern "C" {
                fn _rjem_je_zone_register();
            }

            #[used]
            static _F7: unsafe extern "C" fn() = _rjem_je_zone_register;
        }
    }

    rustc_driver::set_sigpipe_handler();
    rustc_driver::install_ice_hook();

    // When using CI artifacts (with `download_stage1 = true`), tracing is unconditionally built
    // with `--features=static_max_level_info`, which disables almost all rustdoc logging. To avoid
    // this, compile our own version of `tracing` that logs all levels.
    // NOTE: this compiles both versions of tracing unconditionally, because
    // - The compile time hit is not that bad, especially compared to rustdoc's incremental times, and
    // - Otherwise, there's no warning that logging is being ignored when `download_stage1 = true`.
    // NOTE: The reason this doesn't show double logging when `download_stage1 = false` and
    // `debug_logging = true` is because all rustc logging goes to its version of tracing (the one
    // in the sysroot), and all of rustdoc's logging goes to its version (the one in Cargo.toml).
    rustdoc::init_logging();
    rustc_driver::init_env_logger("FUZZ-TARGET-GENERATOR");

    let exit_code = rustc_driver::catch_with_exit_code(|| match rustdoc::get_args() {
        Some(args) => fuzz_target_generator_main_args(&args),
        _ => Err(ErrorReported),
    });

    println!(
        "Fuzz Target Generator exits successfully. Total time cost: {:?} ms",
        start.elapsed().as_millis()
    );

    process::exit(exit_code);
}

type MainResult = Result<(), ErrorReported>;

pub fn fuzz_target_generator_main_args(at_args: &[String]) -> MainResult {
    let args = rustc_driver::args::arg_expand_all(at_args);

    let mut options = getopts::Options::new();
    for option in rustdoc::opts() {
        (option.apply)(&mut options);
    }
    let matches = match options.parse(&args[1..]) {
        Ok(m) => m,
        Err(err) => {
            early_error(ErrorOutputType::default(), &err.to_string());
        }
    };

    // Note that we discard any distinction between different non-zero exit
    // codes from `from_matches` here.
    let options = match rdConfig::Options::from_matches(&matches) {
        Ok(opts) => opts,
        Err(code) => {
            return if code == 0 {
                Ok(())
            } else {
                Err(ErrorReported)
            }
        }
    };
    rustc_interface::util::setup_callbacks_and_run_in_thread_pool_with_globals(
        options.edition,
        1, // this runs single-threaded, even in a parallel compiler
        &None,
        move || fuzz_target_generator_main_options(options),
    )
}

pub fn fuzz_target_generator_main_options(options: rdConfig::Options) -> MainResult {
    let diag = core::new_handler(options.error_format, None, &options.debugging_opts);

    match (options.should_test, options.markdown_input()) {
        (true, true) => return rustdoc::wrap_return(&diag, rustdoc::markdown::test(options)),
        (true, false) => return doctest::run(options),
        (false, true) => {
            return rustdoc::wrap_return(
                &diag,
                rustdoc::markdown::render(&options.input, options.render_options, options.edition),
            );
        }
        (false, false) => {}
    }

    // need to move these items separately because we lose them by the time the closure is called,
    // but we can't create the Handler ahead of time because it's not Send
    let show_coverage = options.show_coverage;
    let run_check = options.run_check;

    // First, parse the crate and extract all relevant information.
    info!("starting to run rustc");

    // Interpret the input file as a rust source file, passing it through the
    // compiler all the way through the analysis passes. The rustdoc output is
    // then generated from the cleaned AST of the crate. This runs all the
    // plug/cleaning passes.
    let crate_version = options.crate_version.clone();

    let output_format = options.output_format;
    // FIXME: fix this clone (especially render_options)
    let externs = options.externs.clone();
    let render_options = options.render_options.clone();
    let scrape_examples_options = options.scrape_examples_options.clone();
    let config = core::create_config(options);

    interface::create_compiler_and_run(config, |compiler| {
        compiler.enter(|queries| {
            let sess = compiler.session();

            if sess.opts.describe_lints {
                let (_, lint_store) = &*queries.register_plugins()?.peek();
                describe_lints(sess, lint_store, true);
                return Ok(());
            }

            // We need to hold on to the complete resolver, so we cause everything to be
            // cloned for the analysis passes to use. Suboptimal, but necessary in the
            // current architecture.
            let resolver = core::create_resolver(externs, queries, sess);

            if sess.has_errors() {
                sess.fatal("Compilation failed, aborting rustdoc");
            }

            let mut global_ctxt = abort_on_err(queries.global_ctxt(), sess).peek_mut();

            global_ctxt.enter(|tcx| {
                let (krate, mut ctxt) = sess.time("run_global_ctxt", || {
                    fuzz_target_generator_run_global_ctxt(
                        tcx,
                        resolver,
                        show_coverage,
                        render_options,
                        output_format,
                    )
                });
                // let render_opts = ctxt.render_options;
                info!("finished with rustc");

                if let Some(options) = scrape_examples_options {
                    return scrape_examples::run(krate, ctxt.render_options, ctxt.cache, tcx, options);
                }

                ctxt.cache.crate_version = crate_version;

                if show_coverage {
                    // if we ran coverage, bail early, we don't need to also generate docs at this point
                    // (also we didn't load in any of the useful passes)
                    return Ok(());
                } else if run_check {
                    // Since we're in "check" mode, no need to generate anything beyond this point.
                    return Ok(());
                }

                info!("going to format");
                match output_format {
                    rdConfig::OutputFormat::Html => sess.time("render_html", || {
                        fuzz_target_run_renderer::<html::render::Context<'_>>(
                            krate,
                            ctxt,
                            tcx,
                        )
                    }),
                    rdConfig::OutputFormat::Json => sess.time("render_json", || {
                        fuzz_target_run_renderer::<json::JsonRenderer<'_>>(
                            krate,
                            ctxt,
                            tcx,
                        )
                    }),
                }
            })
        })
    })
}

pub fn fuzz_target_generator_run_global_ctxt(
    tcx: TyCtxt<'_>,
    resolver: Rc<RefCell<interface::BoxedResolver>>,
    show_coverage: bool,
    render_options: RenderOptions,
    output_format: OutputFormat,
) -> (clean::Crate, DocContext<'_>) {
    // Certain queries assume that some checks were run elsewhere
    // (see https://github.com/rust-lang/rust/pull/73566#issuecomment-656954425),
    // so type-check everything other than function bodies in this crate before running lints.

    // NOTE: this does not call `tcx.analysis()` so that we won't
    // typeck function bodies or run the default rustc lints.
    // (see `override_queries` in the `config`)

    // HACK(jynelson) this calls an _extremely_ limited subset of `typeck`
    // and might break if queries change their assumptions in the future.

    // NOTE: This is copy/pasted from typeck/lib.rs and should be kept in sync with those changes.
    tcx.sess.time("item_types_checking", || {
        tcx.hir()
            .for_each_module(|module| tcx.ensure().check_mod_item_types(module))
    });
    tcx.sess.abort_if_errors();
    tcx.sess.time("missing_docs", || {
        rustc_lint::check_crate(tcx, rustc_lint::builtin::MissingDoc::new);
    });
    tcx.sess.time("check_mod_attrs", || {
        tcx.hir()
            .for_each_module(|module| tcx.ensure().check_mod_attrs(module))
    });
    rustc_passes::stability::check_unused_or_stable_features(tcx);

    let access_levels = AccessLevels {
        map: tcx
            .privacy_access_levels(())
            .map
            .iter()
            .map(|(k, v)| (k.to_def_id(), *v))
            .collect(),
    };

    let mut ctxt = DocContext {
        tcx,
        resolver,
        param_env: ParamEnv::empty(),
        external_traits: Default::default(),
        active_extern_traits: Default::default(),
        substs: Default::default(),
        impl_trait_bounds: Default::default(),
        generated_synthetics: Default::default(),
        auto_traits: tcx
            .all_traits()
            .filter(|&trait_def_id| tcx.trait_is_auto(trait_def_id))
            .collect(),
        module_trait_cache: FxHashMap::default(),
        cache: Cache::new(access_levels, render_options.document_private),
        inlined: FxHashSet::default(),
        output_format,
        render_options,
    };

    // Small hack to force the Sized trait to be present.
    //
    // Note that in case of `#![no_core]`, the trait is not available.
    if let Some(sized_trait_did) = ctxt.tcx.lang_items().sized_trait() {
        let mut sized_trait = clean::inline::build_external_trait(&mut ctxt, sized_trait_did);
        sized_trait.is_auto = true;
        ctxt.external_traits.borrow_mut().insert(
            sized_trait_did,
            TraitWithExtraInfo {
                trait_: sized_trait,
                is_notable: false,
            },
        );
    }

    let hir = tcx.hir();
    let mut visitor = ApiDependencyVisitor::new();
    hir.visit_all_item_likes(&mut visitor);
    // let krate = hir.krate();
    // krate.visit_all_item_likes(&mut visitor);

    debug!("crate: {:?}", tcx.hir().krate());

    // bohao modify
    let mut krate = tcx.sess.time("clean_crate", || clean::krate(&mut ctxt));

    if krate
        .module
        .doc_value()
        .map(|d| d.is_empty())
        .unwrap_or(true)
    {
        let help = format!(
            "The following guide may be of use:\n\
            {}/rustdoc/how-to-write-documentation.html",
            clean::utils::DOC_RUST_LANG_ORG_CHANNEL
        );
        tcx.struct_lint_node(
            rustdoc::lint::MISSING_CRATE_LEVEL_DOCS,
            DocContext::as_local_hir_id(tcx, krate.module.def_id).unwrap(),
            |lint| {
                let mut diag =
                    lint.build("no documentation found for this crate's top-level module");
                diag.help(&help);
                diag.emit();
            },
        );
    }

    pub fn report_deprecated_attr(name: &str, diag: &rustc_errors::Handler, sp: Span) {
        let mut msg =
            diag.struct_span_warn(sp, &format!("the `#![doc({})]` attribute is deprecated", name));
        msg.note(
            "see issue #44136 <https://github.com/rust-lang/rust/issues/44136> \
            for more information",
        );

        if name == "no_default_passes" {
            msg.help("`#![doc(no_default_passes)]` no longer functions; you may want to use `#![doc(document_private_items)]`");
        } else if name.starts_with("passes") {
            msg.help("`#![doc(passes = \"...\")]` no longer functions; you may want to use `#![doc(document_private_items)]`");
        } else if name.starts_with("plugins") {
            msg.warn("`#![doc(plugins = \"...\")]` no longer functions; see CVE-2018-1000622 <https://nvd.nist.gov/vuln/detail/CVE-2018-1000622>");
        }

        msg.emit();
    }

    // Process all of the crate attributes, extracting plugin metadata along
    // with the passes which we are supposed to run.
    for attr in krate.module.attrs.lists(sym::doc) {
        let diag = ctxt.sess().diagnostic();

        let name = attr.name_or_empty();
        // `plugins = "..."`, `no_default_passes`, and `passes = "..."` have no effect
        if attr.is_word() && name == sym::no_default_passes {
            report_deprecated_attr("no_default_passes", diag, attr.span());
        } else if attr.value_str().is_some() {
            match name {
                sym::passes => {
                    report_deprecated_attr("passes = \"...\"", diag, attr.span());
                }
                sym::plugins => {
                    report_deprecated_attr("plugins = \"...\"", diag, attr.span());
                }
                _ => (),
            }
        }


        if attr.is_word() && name == sym::document_private_items {
            ctxt.render_options.document_private = true;
        }
    }

    info!("Executing passes");

    for p in passes::defaults(show_coverage) {
        let run = match p.condition {
            Always => true,
            WhenDocumentPrivate => ctxt.render_options.document_private,
            WhenNotDocumentPrivate => !ctxt.render_options.document_private,
            WhenNotDocumentHidden => !ctxt.render_options.document_hidden,
        };
        if run {
            debug!("running pass {}", p.pass.name);
            krate = tcx.sess.time(p.pass.name, || (p.pass.run)(krate, &mut ctxt));
        }
    }

    if tcx.sess.diagnostic().has_errors_or_lint_errors() {
        rustc_errors::FatalError.raise();
    }

    krate = tcx.sess.time("create_format_cache", || Cache::populate(&mut ctxt, krate)); // commit based on 27th Nov 2021 version
    (krate, ctxt) // commit based on 27th Nov 2021 version
}

fn fuzz_target_run_renderer<'tcx, T: formats::FormatRenderer<'tcx>>(
    krate: clean::Crate,
    // renderopts: rdConfig::RenderOptions,
    // cache: formats::cache::Cache,
    ctxt: DocContext<'tcx>,
    tcx: TyCtxt<'tcx>,
) -> MainResult {
    match fuzz_target_run_format::<T>(krate, ctxt, tcx) {
        Ok(_) => Ok(()),
        Err(e) => {
            let mut msg = tcx
                .sess
                .struct_err(&format!("couldn't generate documentation: {}", e.error));
            let file = e.file.display().to_string();
            if file.is_empty() {
                msg.emit()
            } else {
                msg.note(&format!("failed to create or modify \"{}\"", file))
                    .emit()
            }
            Err(ErrorReported)
        }
    }
}

pub fn fuzz_target_run_format<'tcx, T: renderer::FormatRenderer<'tcx>>(
    krate: clean::Crate,
    // options: RenderOptions,
    // cache: Cache,
    mut ctxt: DocContext<'tcx>,
    tcx: TyCtxt<'tcx>,
) -> Result<(), rustdoc::error::Error> {
    let prof = &tcx.sess.prof;

    let emit_crate = ctxt.render_options.should_emit_crate();

    let start_time = SystemTime::now();
    // 开始构建API DEPENDENCY GRAPH
    let mut api_dependency_graph = api_graph::ApiGraph::new(&krate.name(tcx).to_ident_string());

    // 初始化graph的结构（预设数据结构）
    api_dependency_graph.init_preluded_structure();
    
    //从cache中提出def_id与full_name的对应关系，存入full_name_map来进行调用
    //同时提取impl块中的内容，存入api_dependency_graph
    let mut def_id_map = impl_util::DefIdMap::new();
    // impl_util::extract_impls_from_cache(&ctxt.cache, &tcx, &mut full_name_map, &mut api_dependency_graph);
    impl_util::extract_data_from_ctxt(&mut ctxt, &mut def_id_map, &mut api_dependency_graph);
    
    use crate::rustdoc::formats::FormatRenderer;
    let (mut format_renderer, krate) = prof
        .extra_verbose_generic_activity("create_renderer", T::descr())
        .run(|| Context::init(krate, ctxt.render_options, ctxt.cache, tcx))?;

    if !emit_crate {
        return Ok(());
    }

    // Render the crate documentation
    // let mut work = vec![(format_renderer.make_child_renderer(), krate.module)];
    let mut work = vec![(format_renderer.make_child_renderer(), krate.module)];

    let unknown = Symbol::intern("<unknown item>");
    while let Some((mut cx, item)) = work.pop() {
        if item.is_mod() && T::RUN_ON_MODULE {
            // modules are special because they add a namespace. We also need to
            // recurse into the items of the module as well.
            let _timer =
                prof.generic_activity_with_arg("render_mod_item", item.name.unwrap().to_string());

            fuzz_target_mod_item_in(&mut cx, &item, &mut api_dependency_graph)?;
            let module = match *item.kind {
                clean::StrippedItem(box clean::ModuleItem(m)) | clean::ModuleItem(m) => m,
                _ => unreachable!(),
            };
            for it in module.items {
                debug!("Adding {:?} to worklist", it.name);
                work.push((cx.make_child_renderer(), it));
            }

            cx.mod_item_out()?;
            // FIXME: checking `item.name.is_some()` is very implicit and leads to lots of special
            // cases. Use an explicit match instead.
        } else if item.name.is_some() && !item.is_extern_crate() {
            // prof.generic_activity_with_arg("render_item", &*item.name.unwrap_or(unknown).as_str())
            //     .run(|| cx.item(item))?;
            //item是函数,将函数添加到api_dependency_graph里面去
            let item_type = item.type_();
            if item_type == ItemType::Function {
                let full_name = full_path(&cx, &item);
                //println!("full_name = {}", full_name);
                let def_id = item.def_id;
                match *item.kind {
                    clean::FunctionItem(ref func) => {
                        // public method
                        //println!("func = {:?}",func);
                        let decl = func.decl.clone();
                        let clean::FnDecl { inputs, output, .. } = decl;
                        println!("\n>>>> extract method {} generic info", full_name);
                        let generics = impl_util::extract_generic_info(&ctxt.tcx, &mut def_id_map.trait_name_map, &func.generics, &HashMap::default(), full_name.clone());
                        println!("extract generic info finished <<<<\n");
                        let inputs = api_util::_extract_input_types(&inputs);
                        let output = api_util::_extract_output_type(&output);
                        let api_unsafety =
                            api_function::ApiUnsafety::_get_unsafety_from_fnheader(&func.header);
                        let api_fun = api_function::ApiFunction {
                            full_name,
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
                        };
                        api_dependency_graph.add_api_function(api_fun);
                    }
                    _ => {
                    }
                }
            }
        } else {
            println!("other iterm kind: {:?}", item.kind);
        }
    }

    // extract info from tcx
    // impl_util::extract_info_from_tcx(&tcx, &mut api_dependency_graph);
    // println!("api graph:\n{:#?}", api_dependency_graph);
    //根据mod可见性和预包含类型过滤function
    api_dependency_graph.filter_functions();
    //寻找所有依赖，并且构建序列
    api_dependency_graph.find_all_dependencies();
    api_dependency_graph.find_structures_related_functions();
    print_message::_print_pretty_dependencies(&api_dependency_graph);
    // println!("api dependencies: {:#?}", api_dependency_graph.api_dependencies);
    api_dependency_graph.set_weight();
    // println!("api dependency graph: {:#?}\n" ,api_dependency_graph);
    match env::var("Strategy") {
        Ok(value) => {
            match value.as_str() {
                "bfs" => api_dependency_graph.default_generate_sequences(),
                "unsafe" => api_dependency_graph.unsafe_generate_sequences(),
                "random" => api_dependency_graph.generate_all_possible_sequences(api_graph::GraphTraverseAlgorithm::RandomWalk),
                "ubfs" => api_dependency_graph.unsafe_bfs_generate_sequences(),
                "wubfs" => api_dependency_graph.weighted_unsafe_bfs_generate_sequences(),
                "pattern" => api_dependency_graph.pattern_based_generate_sequences(),
                "dfs" => api_dependency_graph.dfs_generate_sequences(),
                _ => api_dependency_graph.default_generate_sequences(),
            }
        },
        Err(_) => api_dependency_graph.default_generate_sequences(),
    };
    println!("-----------AFTER SEARCH-----------");
    let temp_sequences = api_dependency_graph.api_sequences.clone();
    for i in 0..temp_sequences.len() {
        // 分析语法是否有问题
        let j = temp_sequences.len() - i - 1;
        match api_dependency_graph.sequence_syntax_analyse(&temp_sequences[j]) {
            true => { },
            false => {
                println!("[syntax error] remove sequence {}", temp_sequences[j]);
                api_dependency_graph.api_sequences.remove(j);
                continue;
            }
        }
        match api_dependency_graph.check_generic_refered(&temp_sequences[j]) {
            true => { },
            false => {
                println!("[refer error] remove sequence {}", temp_sequences[j]);
                // api_dependency_graph.api_sequences.remove(j);
            }
        }
        // println!("Seq{}: {:#?}", i, api_dependency_graph.api_sequences[i]);
        // println!("{:#?}\n", api_dependency_graph.api_sequences[i]);
    }
    match env::var("SHOW_ADG"){
        Ok(value) => {
            match value.as_str(){
                "true" => {
                    println!("API DEPENDENCY GRAPH:\n{:#?}", api_dependency_graph);
                },
                _ => {},
            }
        },
        Err(_) => {},
    }
    let random_strategy = false;
    // if !random_strategy {
    //     api_dependency_graph.default_generate_sequences();
    // } else {
    //     api_dependency_graph
    //         .generate_all_possible_sequences(api_graph::GraphTraverseAlgorithm::RandomWalk);
    // }

    // println!("API DEPENDENCY GRAPH:\n{:#?}", api_dependency_graph);
    // use crate::html::afl_util;
    // afl_util::_AflHelpers::_print_all();
    println!("\n>>>>> FUNCTION INDEX LIST");
    let mut generic_count = 0;
    let mut mutate_count = 0;
    let mut drop_count = 0;
    let mut getrawptr_count = 0;
    let mut unsafe_count = 0;
    for i in 0..api_dependency_graph.api_functions.len() {
        let func = &api_dependency_graph.api_functions[i];
        println!("API FUNCTION {}: {}, GENERIC:{}", i, func, func._is_generic_function());
        if func._is_generic_function() {
            generic_count += 1;
        }
        if !func.mutate_info.is_empty() {
            mutate_count += 1;
        }
        if !func.drop_info.is_empty() {
            drop_count += 1;
        }
        if !func.rawptr_info.is_empty(){
            getrawptr_count += 1;
        }
        if !func.unsafe_info.is_empty() | func._unsafe_tag._is_unsafe() {
            unsafe_count += 1;
        }
    }
    println!("------------------------------------------------------------------------------------------------------------------------------------------------");
    println!("|\t{}\t|\t{}\t|\t{}\t|\t{}\t|\t{}\t|\t{}\t|", 
        "TOTAL", 
        "UNSAFE", 
        "RAWPTR", 
        "DROP", 
        "MUTATE", 
        "GENERIC"
    );
    println!("|\t{}\t|\t{}\t|\t{}\t|\t{}\t|\t{}\t|\t{}\t|", 
        api_dependency_graph.api_functions.len(), 
        unsafe_count, 
        getrawptr_count,
        drop_count,
        mutate_count,
        generic_count,
    );
    println!("------------------------------------------------------------------------------------------------------------------------------------------------");
    println!("FUNCTION INDEX LIST <<<<<\n");
    // whether to use random strategy
    // choose sequence in new function:
    println!("\n>>>>> WRITING FILE START");
    let file_helper = file_util::FileHelper::new(&api_dependency_graph, random_strategy);
    //println!("file_helper:{:?}", file_helper);
    file_helper.write_files();
    file_helper.write_libfuzzer_files();
    println!("WRITING FILE FINISHED <<<<<\n");
    // println!("Parameters: {:#?}", api_dependency_graph.api_parameters);
    println!("Generic functions Num: {:#?}", generic_count);
    // if api_dependency_graph.generic_functions.len() > api_dependency_graph.api_functions.len() {
    //     println!("{:#?}", api_dependency_graph.generic_functions);
    // }
    println!("total functions in crate : {:?}", api_dependency_graph.api_functions.len());
    println!("total test sequences : {:?}", api_dependency_graph.api_sequences.len());
    println!("Total Time Used: {:?}", SystemTime::now().duration_since(start_time).unwrap().as_millis());
    
    // for param in &api_dependency_graph.api_parameters {
    //     println!("{:#?}", param);
    // }
    prof.extra_verbose_generic_activity("renderer_after_krate", T::descr())
        .run(|| format_renderer.after_krate())
}

fn fuzz_target_mod_item_in(
    cx: &mut Context<'_>,
    item: &clean::Item,
    api_dependency_graph: &mut api_graph::ApiGraph,
) -> Result<(), Error> {
    // Stripped modules survive the rustdoc passes (i.e., `strip-private`)
    // if they contain impls for public types. These modules can also
    // contain items such as publicly re-exported structures.
    //
    // External crates will provide links to these structures, so
    // these modules are recursed into, but not rendered normally
    // (a flag on the context).
    if !cx.render_redirect_pages {
        cx.render_redirect_pages = item.is_stripped();
    }
    let scx = &cx.shared;
    let item_name = item.name.as_ref().unwrap().to_string();
    cx.dst.push(&item_name);
    cx.current.push(item_name);

    // bohao modify
    let mod_name = cx.current.join("::");
    api_dependency_graph.add_mod_visibility(&mod_name, &item.visibility);

    info!("Recursing into {}", cx.dst.display());

    let buf = cx.render_item(item, true);
    // buf will be empty if the module is stripped and there is no redirect for it
    if !buf.is_empty() {
        cx.shared.ensure_dir(&cx.dst)?;
        let joint_dst = cx.dst.join("index.html");
        scx.fs.write(joint_dst, buf)?;
    }

    // Render sidebar-items.js used throughout this module.
    if !cx.render_redirect_pages {
        let module = match *item.kind {
            clean::StrippedItem(box clean::ModuleItem(ref m)) | clean::ModuleItem(ref m) => m,
            _ => unreachable!(),
        };
        let items = cx.build_sidebar_items(module);
        let js_dst = cx.dst.join("sidebar-items.js");
        let v = format!(
            "initSidebarItems({});",
            serde_json::to_string(&items).unwrap()
        );
        scx.fs.write(js_dst, v)?;
    }
    Ok(())
}

fn full_path(cx: &Context<'_>, item: &clean::Item) -> String {
    let mut s = cx.current.join("::");
    s.push_str("::");
    s.push_str(&item.name.as_ref().unwrap().to_ident_string());
    s
}
