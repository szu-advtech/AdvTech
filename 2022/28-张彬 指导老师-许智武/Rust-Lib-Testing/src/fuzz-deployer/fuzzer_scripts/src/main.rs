#[macro_use]
extern crate lazy_static;
extern crate regex;
extern crate config;


use std::{process::{Command, exit, Output, Stdio}};
use std::env;
use std::collections::{HashMap, HashSet};
use regex::Regex;
use std::str;
use std::path::PathBuf;
use std::fs::{self, OpenOptions, File};
use std::io::{self, Write, Seek, SeekFrom};
use std::io::prelude::*;
use std::io::BufReader;
use std::thread;
use std::time::Duration;
use std::time::Instant;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

lazy_static! {
    static ref CRATES: HashSet<&'static str> = {
        let mut m = HashSet::new();
        //m.insert("url");
        m.insert("regex-syntax");
        m.insert("semver-parser");
        m.insert("bat");
        m.insert("xi-core-lib");
        m.insert("clap"); 
        m.insert("regex");
        m.insert("serde_json");
        m.insert("tui");
        m.insert("semver");
        m.insert("http");
        m.insert("flate2");
        m.insert("smoltcp");
        m.insert("proc-macro2");
        m.insert("time");
        //fudge like crates
        m.insert("fudge_like_url");
        m.insert("fudge_like_regex");
        m.insert("fudge_like_time");

        //fudge crates
        m.insert("fudge_url");
        m.insert("fudge_regex");
        m.insert("fudge_time");

        m
    };
}

lazy_static! {
    static ref INVALID_TARGET_NUMBER: HashMap<&'static str, usize> = {
        let mut m = HashMap::new();
        m.insert("proc-macro2", 1);
        m.insert("clap", 4);
        m.insert("syn", 1);
        m
    };
}

static MULTIPLE_TARGETS: &'static str = "multipleTargets";
static CRASH_DIR: &'static str = "default/crashes";
static TEST_FILE_DIR: &'static str = "test_files";
static REPLAY_FILE_DIR: &'static str = "replay_files";
static AFL_INPUT_DIR: &'static str = "afl_init";
static AFL_OUTPUT_DIR: &'static str = "out";
static CARGO_TOML: &'static str = "Cargo.toml";
static BUILD_SCRIPT: &'static str = "build";
static AFL_DEPENDENCY: &'static str = "afl = \"*\"";
static TMIN_OUTPUT_DIR: &'static str = "tmin_output";
static CMIN_OUTPUT_DIR: &'static str = "cmin_output";
static STATISTIC_OUTPUT_FILE: &'static str = "statistics";
static EDITION: &'static str = "debug";
static EXIT_TIME_DIR: &'static str = "exit_time";
static SHOWMAP_DIR: &'static str = "showmap";

// static mut ERROR_FILE : Vec<String> = Vec::new();

#[derive(Debug,Clone)]
struct UserOptions {
    crate_name: Option<String>,
    find_literal: Option<usize>,
    check: bool,
    clean: bool,
    build: bool,
    fuzz: bool,
    crash: bool,
    prepare: bool, 
    tmin: bool,
    cmin: bool,
    replay: bool,
    statistic: bool,
    showmap: bool,
    init_afl_input: bool,
    all: bool,
}

impl UserOptions {
    fn new() -> Self {
        UserOptions {
            crate_name: None,
            find_literal: None,
            check: false,
            clean: false,
            build: false,
            all: false,
            crash: false,
            fuzz: false,
            prepare: false,
            tmin: false,
            cmin: false,
            statistic: false,
            showmap: false,
            replay: false,
            init_afl_input: false,
        }
    }

    fn new_from_cli(args: Vec<String>) -> Self {
        let mut user_options = UserOptions::new();
        user_options.extract_options(args);
        user_options
    }

    fn extract_options(&mut self, args: Vec<String>) {
        let mut args_iter = args.iter();
        let _ = args_iter.next(); //把程序名字跳过

        let list_option = Regex::new("(-l$|--list)").unwrap();
        let find_literal_option = Regex::new("(-f$|--find-literal)").unwrap();
        let check_option = Regex::new("(-c$|--check)").unwrap();
        let clean_option = Regex::new("-clean").unwrap();
        let build_option = Regex::new("(-b$|--build)").unwrap();
        let fuzz_option = Regex::new("-fuzz").unwrap();
        let all_option = Regex::new("(-a$|--all)").unwrap();
        let help_option = Regex::new("(-h$|--help)").unwrap();
        let crash_option = Regex::new("-crash").unwrap();
        let prepare_option = Regex::new("(-p$|--prepare)").unwrap();
        let tmin_option = Regex::new("(-t$|--tmin)").unwrap();
        let cmin_option = Regex::new("-cmin").unwrap();
        let statistic_option = Regex::new("(-s$|--statistic)").unwrap();
        let showmap_option = Regex::new("-showmap").unwrap();
        let replay_option = Regex::new("(-r$|--replay)").unwrap();
        let init_afl_input_option = Regex::new("(-i$|--init)").unwrap();

        while let Some(s) = args_iter.next(){
            if list_option.is_match(s.as_str()) {
                list_crates();
                exit(0);
            }
            if help_option.is_match(s.as_str()) {
                println!("{}",help_message());
                exit(0);
            }
            if find_literal_option.is_match(s.as_str()) {
                if let Some(input_number) = args_iter.next() {
                    let input_number = input_number.parse::<usize>();
                    if input_number.is_ok() {
                        self.find_literal = Some(input_number.unwrap());
                        continue;
                    }
                }
                println!("Invalid -f/find_literal flag.");
                exit(-1);
            }
            if check_option.is_match(s.as_str()) {
                self.check = true;
                continue;
            }
            if clean_option.is_match(s.as_str()) {
                self.clean = true;
                continue;
            }
            if build_option.is_match(s.as_str()) {
                self.build = true;
                continue;
            }
            if fuzz_option.is_match(s.as_str()) {
                self.fuzz = true;
                continue;
            }
            if all_option.is_match(s.as_str()) {
                self.all = true;
                continue;
            }
            if crash_option.is_match(s.as_str()) {
                self.crash = true;
                continue;
            }
            if prepare_option.is_match(s.as_str()) {
                self.prepare = true;
                continue;
            }
            if tmin_option.is_match(s.as_str()) {
                self.tmin = true;
                continue;
            }
            if cmin_option.is_match(s.as_str()) {
                self.cmin = true;
                continue;
            }
            if replay_option.is_match(s.as_str()) {
                self.replay = true;
                continue;
            }
            if statistic_option.is_match(s.as_str()) {
                self.statistic = true;
                continue;
            }
            if showmap_option.is_match(s.as_str()) {
                self.showmap = true;
                continue;
            }
            if init_afl_input_option.is_match(s.as_str()) {
                self.init_afl_input = true;
                continue;
            }
            if self.crate_name.is_none() {
                self.crate_name = Some(s.clone());
                continue;
            }
            println!("Invalid Options.");
            exit(-1);
        }
        if self.crate_name.is_none() {
            println!("No valid crate is provided.");
            exit(-1);
        }
    } 
}

fn list_crates() {
    for crate_name in CRATES.iter() {
        println!("{}", crate_name);
    }
}

fn help_message() -> &'static str {
"fuzzer_scripts 0.1.0

USAGE: 
    fuzzer_scripts FLAGS CRATE_NAME
        
FLAGS:
    -l,--list           list all supported crates
    -h,--help           print help message
    -f,--find_literal   find literals(example: -f 3 url)
    -c,--check          check precondition
    -clean              clean test directory(may corrupt history data)
    -b,--build          init test directory and build afl test files
    -fuzz               run afl
    -a,--all            clean,build,and fuzz(may corrupt history data)
    -crash              check if any crash was found
    -p,--prepare        prepare test files
    -t,--tmin           use afl tmin to reduce test file size
    -cmin               use afl cmin to reduce test file number
    -r,--replay         replay crash files to check whether it's real crash
    -s,--statistic      output statictic fuzz result info for a crate
    -i,--init           init afl input files for each target
"  
//-showmap            output coverage infomation generated by showmap(showmap is not well designed)
}

fn do_work(user_options: &UserOptions) {
    let crate_name = user_options.crate_name.as_ref().unwrap();
    // crate_name.replace("-", "_");
    if user_options.check {
        check_pre_condition(crate_name);
        println!("check success.");
        exit(0);
    }
    if user_options.find_literal.is_some() {
        do_find_literal(crate_name, user_options.find_literal.unwrap().to_string());
        println!("find literal");
        exit(0);
    }
    if user_options.prepare {
        prepare_test_files();
        println!("prepare test files");
        exit(0);
    }
    if user_options.clean {
        clean(crate_name);
        println!("clean");
        exit(0);
    }
    if user_options.crash {
        println!("find crashes");
        print_crashes(crate_name);
        exit(0);
    }

    if user_options.build {
        let tests = check_pre_condition(crate_name);
        init_test_dir(crate_name, &tests);
        build_afl_tests(crate_name);
        init_afl_input(crate_name);
        check_build(crate_name, &tests);
        exit(0);
    }
    if user_options.fuzz {
        let tests = check_pre_condition(crate_name);
        check_build(crate_name, &tests);
        fuzz_it(crate_name, &tests);
        exit(0);
    }
    if user_options.tmin {
        tmin(crate_name);
        exit(0);
    }
    if user_options.cmin {
        cmin(crate_name);
        exit(0);
    }
    if user_options.replay {
        replay_crashes(crate_name);
        exit(0);
    }
    if user_options.statistic {
        output_statistics(crate_name);
        exit(0);
    }
    if user_options.showmap {
        showmap(crate_name);
        exit(0);
    }
    if user_options.init_afl_input {
        println!("init afl input for each target");
        let tests = check_pre_condition(crate_name);
        check_build(crate_name, &tests);
        init_afl_input(crate_name);
        exit(0);
    }
    if user_options.all {
        let tests = check_pre_condition(crate_name);
        clean(crate_name);
        init_test_dir(crate_name, &tests);
        build_afl_tests(crate_name);
        init_afl_input(crate_name);
        check_build(crate_name, &tests);
        fuzz_it(crate_name, &tests);
        exit(0);
    }
    //default work
    let tests = check_pre_condition(crate_name);
    init_test_dir(crate_name, &tests);
    build_afl_tests(crate_name);
    check_build(crate_name, &tests);
    fuzz_it(crate_name, &tests);
}

fn do_find_literal(crate_name: &String, input_number: String) {
    let input_dir = env::current_dir().unwrap().into_os_string().into_string().unwrap(); // modified by wcventure
    let mut output_dir = env::current_dir().unwrap().into_os_string().into_string().unwrap();
    output_dir.push_str("/fuzz_target/");
    output_dir.push_str(crate_name);
    match env::var("Strategy"){
        Ok(value) => {
            output_dir.push_str(("_".to_owned() + &value).as_str());
        },
        Err(_) => {},
    };
    output_dir.push_str("_fuzz");
    let args = vec!["-i", input_dir.as_str(),"-o",output_dir.as_str(), "-n",input_number.as_str()];
    let output = Command::new("find_literal").args(args).output().unwrap_or_else(
        |_| {println!("find_literal encounter problems."); exit(-1);}
    );
    print_output(output);
}

fn prepare_test_files() {
    let src_dir = env::current_dir().unwrap().into_os_string().into_string().unwrap(); // modified by wcventure
    let src_path = PathBuf::from(src_dir.as_str());
    let output = Command::new("cargo").current_dir(&src_path).arg("clean").output().unwrap();
    print_output(output);
    println!("cargo clean");
    let output = Command::new("cargo").current_dir(&src_path).arg("doc").arg("-v").output().unwrap();
    let stderr = str::from_utf8(output.stderr.as_slice()).unwrap();
    //println!("stderr = {}", stderr);
    let stderr_lines:Vec<&str> = stderr.split("\n").collect();
    let stderr_lines_number = stderr_lines.len();
    if stderr_lines_number < 3 {
        println!("cargo doc goes wrong");
        exit(-1);
    }
    let rustdoc_line = stderr_lines[stderr_lines_number - 3];
    println!("rustdoc line = {}", rustdoc_line);
    let pattern = Regex::new(r#"`rustdoc.+`"#).unwrap();
    let raw_command = pattern.find(rustdoc_line).unwrap().as_str();
    let command = raw_command.replace("rustdoc ", "").replace("`", "");
    let command_args:Vec<&str> = command.split(" ").collect();
    println!("command_args = {:?}", command_args);
    let output = Command::new("fuzz-target-generator").args(command_args).current_dir(&src_dir).output().unwrap();
    print_output(output);
}

pub fn print_output(output: Output) {
    let stdout = &output.stdout;
    if stdout.len() > 0 {
        println!("{}", str::from_utf8(stdout.as_slice()).unwrap());
    }
    let stderr = &output.stderr;
    if stderr.len() > 0 {
        eprintln!("{}", str::from_utf8(stderr.as_slice()).unwrap());
    }
}

pub fn summary_output(outputs: &Vec<Output>, commands: &Vec<String>) {
    println!("------------------- SUMMARY -------------------");
    let panicked_regex = Regex::new(r".*(panicked).*").unwrap();
    let summary_regex = Regex::new(r".*(SUMMARY:).*").unwrap();
    let memory_regex = Regex::new(r".*(memory allocation of).*").unwrap();
    let mut build_id_regex = Regex::new(r"BuildId: ([0-9A-Za-z]){40}").unwrap();
    let mut panicked_result: Vec<&str> = Vec::new();
    let mut summary_result: Vec<&str> = Vec::new();
    let mut memory_result: Vec<&str> = Vec::new();
    // let mut build_id_map = HashMap::new();
    let mut crash_count = 0;
    for i in 0..outputs.len() {
        let output = &outputs[i];
        let stderr = &output.stderr;
        if stderr.len() > 0 {
            let stderr_str = str::from_utf8(stderr.as_slice()).unwrap();
            let mut panicked_mat: Vec<&str> = panicked_regex.find_iter(&stderr_str).map(|x| x.as_str()).collect();
            let mut summary_mat: Vec<&str> = summary_regex.find_iter(&stderr_str).map(|x| x.as_str()).collect();
            let mut memory_mat: Vec<&str> = memory_regex.find_iter(&stderr_str).map(|x| x.as_str()).collect();
            if panicked_mat.len() == 0 && summary_mat.len() == 0 && memory_mat.len() == 0 {
                eprintln!("Miss: {}", stderr_str);
            }
            panicked_result.append(&mut panicked_mat);
            summary_result.append(&mut summary_mat);
            memory_result.append(&mut memory_mat);
            crash_count += 1;
        }
    }
    let panicked_set: HashSet<&str> = HashSet::from_iter(panicked_result);
    let summary_set: HashSet<&str> = HashSet::from_iter(summary_result);
    let memory_set: HashSet<&str> = HashSet::from_iter(memory_result);
    let mut count = 0;
    for panicked in &panicked_set {
        println!("P{}: {}", count+1, panicked);
        count += 1;
    }
    count = 0;
    for summary in &summary_set {
        println!("S{}: {}", count+1, summary);
        count += 1;
    }
    count = 0;
    for memory in &memory_set {
        println!("M{}: {}", count+1, memory);
        count += 1;
    }
    println!("Total Crash: {}", crash_count);
    println!("Total: {}, P: {}, S: {}, M: {}", panicked_set.len()+summary_set.len()+memory_set.len(), panicked_set.len(), summary_set.len(), memory_set.len());
}

//检查一个crate的前置条件是否满足，包括
//test_files, replay_files, afl_init
pub fn check_pre_condition(crate_name: &String) -> Vec<String> {
    check_static();

    let mut crate_test_dir = env::current_dir().unwrap().into_os_string().into_string().unwrap();
    crate_test_dir.push_str("/fuzz_target/");
    crate_test_dir.push_str(crate_name);
    match env::var("Strategy"){
        Ok(value) => {
            crate_test_dir.push_str(("_".to_owned() + &value).as_str());
        },
        Err(_) => {},
    };
    crate_test_dir.push_str("_fuzz");
    let crate_test_path = PathBuf::from(crate_test_dir);
    
    let afl_init_dir = crate_test_path.clone().join(AFL_INPUT_DIR);
    check_no_empty_directory(&afl_init_dir);
    let test_file_dir = crate_test_path.clone().join(TEST_FILE_DIR);
    let test_file_entries = check_no_empty_directory(&test_file_dir);

    let replay_file_dir = crate_test_path.clone().join(REPLAY_FILE_DIR);
    let replay_file_entries = check_no_empty_directory(&replay_file_dir);

    let mut test_filenames = Vec::new();
    check_rs_file(&test_file_entries, &mut test_filenames);
    //println!("{:?}",test_filenames);

    let mut replay_filenames  = Vec::new();
    check_rs_file(&replay_file_entries, &mut replay_filenames);
    //println!("{:?}", replay_filenames);

    for test_file in &test_filenames {
        let replay_file = test_file.clone().replace("test", "replay");
        if !replay_filenames.contains(&replay_file) {
            println!("Not exist replay file for test file {}", test_file);
            exit(-1);
        }
    }
    test_filenames
}

fn check_static() { //TODO: check Cargo.toml and check member
    let path = env::current_dir().unwrap();
    let dir = path.as_path().read_dir().unwrap();
    let mut checkstatus = false;
    for x in dir {
       if let Ok(path) = x {
           //println!("{:?}", path.file_name()); // 该路径下所有文件和文件夹名称
           // 是否存在某个文件
           if path.file_name().eq("Cargo.toml") {
                checkstatus = true;
                //println!("存在 `Cargo.toml` 文件!");
                break;
           }
        }
    }
    if checkstatus == false {
        println!("不存在 `Cargo.toml` 文件!");
        exit(-1)
    }
        
}

fn check_maybe_empty_directory(dir: &PathBuf) -> Vec<PathBuf> {
    if !dir.is_dir() {
        //println!("File directory {:?} not exists.", dir);
        return Vec::new();
    }
    let file_entry = fs::read_dir(dir).unwrap();
    let file_entries = file_entry.map(|res| res.map(|e| e.path())).collect::<Result<Vec<_>, io::Error>>().unwrap();
    file_entries
}

fn check_no_empty_directory(dir: &PathBuf) -> Vec<PathBuf> {
    let file_entries = check_maybe_empty_directory(dir);
    if file_entries.len() == 0 {
        println!("No file in {:?}.", dir);
        exit(-1);
    }
    file_entries
}

fn check_rs_file(file_entries: &Vec<PathBuf>, filenames: &mut Vec<String>) {
    let regex = Regex::new(r"^(\w|_)+.rs$").unwrap();
    for pathbuf in file_entries {
        let last_file = last_file_name(pathbuf);
        if regex.is_match(last_file) && pathbuf.is_file(){
            filenames.push(last_file.to_string().replace(".rs", ""));
        }else {
            println!("Invalid file {} was found.", last_file);
            exit(-1);
        }
    }
}

fn last_file_name(path: &PathBuf) -> &str {
    let filename = path.to_str().unwrap();
    let filename: Vec<&str> = filename.split("/").collect();
    filename.last().unwrap()
}

fn clean(crate_name: &String) {
    let except_files = vec![AFL_INPUT_DIR, REPLAY_FILE_DIR, TEST_FILE_DIR];
    let mut crate_test_dir = env::current_dir().unwrap().into_os_string().into_string().unwrap();
    crate_test_dir.push_str("/fuzz_target/");
    crate_test_dir.push_str(crate_name);
    match env::var("Strategy"){
        Ok(value) => {
            crate_test_dir.push_str(("_".to_owned() + &value).as_str());
        },
        Err(_) => {},
    };
    crate_test_dir.push_str("_fuzz");
    let test_path = PathBuf::from(crate_test_dir);
    let test_path = test_path.join(MULTIPLE_TARGETS);

    let file_entries = check_maybe_empty_directory(&test_path);
    for file_entry in &file_entries {
        if !except_files.contains(&last_file_name(file_entry)) {
            if file_entry.is_dir() {
                fs::remove_dir_all(file_entry).unwrap_or_else(
                    |_| {println!("Encounter error when removing {:?}.", file_entry); exit(-1);}
                );
            }
            if file_entry.is_file() {
                fs::remove_file(file_entry).unwrap_or_else(
                    |_| {println!("Encounter error when removing {:?}.", file_entry); exit(-1);}
                );
            }
        }
    }
}

fn init_test_dir(crate_name: &String, tests:&Vec<String>) {
    let real_crate_name = crate_name.clone().replace("_", "-");
    println!("real crate name: {:#?}", real_crate_name);
    let mut test_dir = env::current_dir().unwrap().into_os_string().into_string().unwrap();
    test_dir.push_str("/fuzz_target/");
    test_dir.push_str(crate_name);
    match env::var("Strategy"){
        Ok(value) => {
            test_dir.push_str(("_".to_owned() + &value).as_str());
        },
        Err(_) => {},
    };
    test_dir.push_str("_fuzz");
    let test_path = PathBuf::from(test_dir);
    let test_path = test_path.join(MULTIPLE_TARGETS);
    println!("TEST PATH: {:?}", test_path);
    //生成输出目录
    let output_dir = test_path.clone().join(AFL_OUTPUT_DIR);
    ensure_empty_dir(&output_dir);
    // fs::create_dir_all(&output_dir).unwrap_or_else(
    //     |_| {println!("Encounter error when creating {:?}.", output_dir); exit(-1);}
    // );

    //生成cargo.toml内容
    //根据环境变量NewCargoToml判断是否生成toml文件
    match env::var("NewCargoToml") {
        Ok(value) =>{
            match ("_".to_owned() + &value).as_str(){
                "false" => {
                    println!("use old cargo.toml");
                },
                _ => {
                    let cargo_toml_path = test_path.clone().join(CARGO_TOML);
                    let mut cargo_toml_file = fs::File::create(&cargo_toml_path).unwrap_or_else(
                        |_| {println!("Encounter error when creating {:?}.", cargo_toml_path); exit(-1);}
                    );
                    let cargo_content = cargo_workspace_file_content(tests);
                    cargo_toml_file.write_all(cargo_content.as_bytes()).unwrap_or_else(
                        |_| {println!("write file {:?} failed.", cargo_toml_file); exit(-1);}
                    );
                    println!("cargo.toml created");
                }
            }
        },
        Err(_) => {
            let cargo_toml_path = test_path.clone().join(CARGO_TOML);
            let mut cargo_toml_file = fs::File::create(&cargo_toml_path).unwrap_or_else(
                |_| {println!("Encounter error when creating {:?}.", cargo_toml_path); exit(-1);}
            );
            let cargo_content = cargo_workspace_file_content(tests);
            cargo_toml_file.write_all(cargo_content.as_bytes()).unwrap_or_else(
                |_| {println!("write file {:?} failed.", cargo_toml_file); exit(-1);}
            );
            println!("cargo.toml created");
        }
    }

    //对于每个test_file新建项目
    for test in tests {
        println!("Initializing {:?}", test);
        let test_cargo_path = test_path.clone().join(test);
        let _ = Command::new("cargo").arg("new").arg(test_cargo_path.as_os_str()).output().unwrap();
    }

    //对于每个replay_file新建项目
    let mut replays = Vec::new();
    for test in tests {
        let replay = test.clone().replace("test", "replay");
        println!("Initializing {:?}", replay);
        let replay_cargo_path = test_path.clone().join(&replay);
        replays.push(replay);
        let _ = Command::new("cargo").arg("new").arg(replay_cargo_path.as_os_str()).output().unwrap();
    }

    //生成build script(貌似没必要)
    let build_script_path = test_path.clone().join(BUILD_SCRIPT);
    let mut build_script_file = fs::File::create(&build_script_path).unwrap_or_else(
        |_| {println!("Encounter error when creating {:?}.", build_script_path); exit(-1);}
    );
    let build_script = build_script_content(&test_path);
    build_script_file.write_all(build_script.as_bytes()).unwrap_or_else(
        |_| {println!("write file {:?} failed.", build_script_file); exit(-1);}
    );
    Command::new("chmod").arg("+x").arg(build_script_path.as_os_str()).output().unwrap();

    //判断source_crate的Cargo.toml中的第一行是否是"[workspace]"
    let source_crate_dir = env::current_dir().unwrap().into_os_string().into_string().unwrap();
    let source_crate_path = PathBuf::from(source_crate_dir);
    let source_crate_cargo_toml_path = source_crate_path.clone().join(CARGO_TOML);
    let mut does_contain_workspace_member = false;
    let source_crate_toml_file = File::open(&source_crate_cargo_toml_path).unwrap();
    let reader = BufReader::new(source_crate_toml_file);
    let mut break_count = 0; // 记录查看的行数
    for line in reader.lines() {
        let line = line.unwrap(); // line 是 std::result::Result<std::string::String, std::io::Error> 类型, line 不包含换行符
        
        println!("[CARGO_TOML] {}", line); // 判断当前行是否为"[workspace]"
        if line.as_str().trim() == "[workspace]" {
            does_contain_workspace_member = true;
            break;
        }

        break_count = break_count + 1;
        if break_count > 5 { // 只看前5行
            break;
        }
    }
    //println!("{}", text);

    //为每个test crate添加依赖
    for test in tests {
        let test_cargo_toml_path = test_path.clone().join(test).join(CARGO_TOML);
        let mut file = OpenOptions::new().read(true).write(true).open(&test_cargo_toml_path).unwrap_or_else(
            |_| {println!("can't open file {:?}.", test_cargo_toml_path); exit(-1);}
        );

        //讲游标移动到"[dependencies]"的位置
        let mut reader = BufReader::new(&file);
        for line in reader.by_ref().lines() {
            let line = line.unwrap(); // line 是 std::result::Result<std::string::String, std::io::Error> 类型, line 不包含换行符
            if line.as_str().trim() == "[dependencies]" {
                break;
            }
        }
        let current_pos = reader.seek (SeekFrom::Current (0)).expect ("Could not get current position!");
        file.seek(std::io::SeekFrom::Start(current_pos)).unwrap();


        file.write_all(AFL_DEPENDENCY.as_bytes()).unwrap_or_else(
            |_| {println!("write file {:?} failed.", test_cargo_toml_path); exit(-1);}
        );
        file.write_all("\n".as_bytes()).unwrap();
        let curdir = env::current_dir().unwrap().into_os_string().into_string().unwrap();
        let mut version = String::from("{path=\"");
        version.push_str(curdir.as_str());
        
        if does_contain_workspace_member {
            version.push_str("/");
            version.push_str(crate_name);
        }
        version.push_str("\"}");
        let crate_dependency = format!("{} = {}\n", crate_name, version);
        file.write_all(crate_dependency.as_bytes()).unwrap_or_else(
            |_| {println!("write file {:?} failed.", test_cargo_toml_path); exit(-1);}
        );
    }

    //为每个replay crate添加依赖
    for replay in &replays {
        let replay_cargo_toml_path = test_path.clone().join(replay).join(CARGO_TOML);
        let mut file = OpenOptions::new().read(true).write(true).open(&replay_cargo_toml_path).unwrap_or_else(
            |_| {println!("can't open file {:?}.", replay_cargo_toml_path); exit(-1);}
        );
        let curdir = env::current_dir().unwrap().into_os_string().into_string().unwrap();
        let mut version = String::from("{path=\"");
        version.push_str(curdir.as_str());
        
        if does_contain_workspace_member {
            version.push_str("/");
            version.push_str(crate_name);
        }
        version.push_str("\"}");

        let crate_dependency = format!("{} = {}\n", crate_name, version);

        //讲游标移动到"[dependencies]"的位置
        let mut reader = BufReader::new(&file);
        for line in reader.by_ref().lines() {
            let line = line.unwrap(); // line 是 std::result::Result<std::string::String, std::io::Error> 类型, line 不包含换行符
            if line.as_str().trim() == "[dependencies]" {  
                break;
            }
        }
        let current_pos = reader.seek (SeekFrom::Current (0)).expect ("Could not get current position!");
        file.seek(std::io::SeekFrom::Start(current_pos)).unwrap();
        
        file.write_all(crate_dependency.as_bytes()).unwrap_or_else(
            |_| {println!("write file {:?} failed.", replay_cargo_toml_path); exit(-1);}
        );
    }

    let mut ori_test_dir = env::current_dir().unwrap().into_os_string().into_string().unwrap();
    ori_test_dir.push_str("/fuzz_target/");
    ori_test_dir.push_str(crate_name);
    match env::var("Strategy"){
        Ok(value) => {
            ori_test_dir.push_str(("_".to_owned() + &value).as_str());
        },
        Err(_) => {},
    };
    ori_test_dir.push_str("_fuzz");
    let ori_test_path = PathBuf::from(ori_test_dir);

    //复制测试文件
    for test in tests {
        let to_path = test_path.clone().join(test).join("src").join("main.rs");
        let mut test_name = test.clone();
        test_name.push_str(".rs");
        let from_path = ori_test_path.clone().join(TEST_FILE_DIR).join(test_name);
        Command::new("cp").arg(from_path.as_os_str()).arg(to_path.as_os_str()).output().unwrap();
    }

    //复制replay文件
    for replay in &replays {
        let to_path = test_path.clone().join(replay).join("src").join("main.rs");
        let mut replay_name = replay.clone();
        replay_name.push_str(".rs");
        let from_path = ori_test_path.clone().join(REPLAY_FILE_DIR).join(replay_name);
        Command::new("cp").arg(from_path.as_os_str()).arg(to_path.as_os_str()).output().unwrap();
    }
}

fn cargo_workspace_file_content(tests:&Vec<String>) -> String {
    let mut content = format!("[workspace]\nmembers = [\n");
    for test in tests {
        let one_test = format!("\t\"{}\",\n", test);
        content.push_str(one_test.as_str());
        let one_replay = one_test.replace("test", "replay");
        content.push_str(one_replay.as_str());
    }
    content.push_str("]\n");
    content
}

fn build_script_content(test_path: &PathBuf) -> String {
    format!("cd {:?}
cargo afl build
cd -", test_path).replace("\"", "")
}

// cargo afl build --offline
fn build_afl_tests(crate_name: &String) {
    println!("Building {:?}", crate_name);
    let mut test_dir = env::current_dir().unwrap().into_os_string().into_string().unwrap();
    test_dir.push_str("/fuzz_target/");
    test_dir.push_str(crate_name);
    match env::var("Strategy"){
        Ok(value) => {
            test_dir.push_str(("_".to_owned() + &value).as_str());
        },
        Err(_) => {},
    };
    test_dir.push_str("_fuzz");
    let test_path = PathBuf::from(test_dir);
    let test_path = test_path.join(MULTIPLE_TARGETS);
    let error_regex = Regex::new(r"(error:).*`.*\n").unwrap();
    let cargo_toml_path = test_path.clone().join(CARGO_TOML);
    let mut cargo_toml_file = fs::File::open(&cargo_toml_path).unwrap();
    let mut cargo_toml_content = String::new();
    cargo_toml_file.read_to_string(&mut cargo_toml_content).unwrap();
    loop {
        let output = Command::new("cargo").arg("+nightly").arg("afl").arg("build").arg("--offline").current_dir(test_path.as_os_str()).output().unwrap();
        println!("status: {}", output.status); // 0代表成功
        if output.status.success() {
            println!("Finished building crate.");
            break;
        }
        // println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        // println!("stderr: {:#?}", String::from_utf8_lossy(&output.stderr));
        
        let output_stderr = String::from_utf8_lossy(&output.stderr);
        let mat_results: Vec<&str> = error_regex.find_iter(&output_stderr).map(|x| x.as_str()).collect();
        println!("Compile error found");
        let mat = mat_results[0];
        let mat_name = {
            let start_index = mat.find('`').unwrap() + 1;
            let end_index = mat.rfind('`').unwrap();
            &mat[start_index..end_index]
        };
        let delete_name1 = format!("\t\"{}\",\n", mat_name);
        let delete_name2 = delete_name1.replace("replay", "test");
        print!("Delete error file in Cargo.toml: {}", delete_name1);
        println!("Delete error file in Cargo.toml: {}", delete_name2);
        cargo_toml_content = cargo_toml_content.replace(&delete_name1, "");
        cargo_toml_content = cargo_toml_content.replace(&delete_name2, "");
        fs::File::create(&cargo_toml_path).unwrap().write_all(&cargo_toml_content.as_bytes()).unwrap();
    }
}

fn check_build(crate_name: &String, tests:&Vec<String>) {
    let mut test_dir = env::current_dir().unwrap().into_os_string().into_string().unwrap();
    test_dir.push_str("/fuzz_target/");
    test_dir.push_str(crate_name);
    match env::var("Strategy"){
        Ok(value) => {
            test_dir.push_str(("_".to_owned() + &value).as_str());
        },
        Err(_) => {},
    };
    test_dir.push_str("_fuzz");
    let test_path = PathBuf::from(test_dir);
    let test_path = test_path.join(MULTIPLE_TARGETS);
    let target_path = test_path.join("target").join(EDITION);

    // let mut flag = true;
    let mut success_count = 0;
    let mut failed_count = 0;
    for test in tests {
        let build_afl_file_path = target_path.join(test);
        if !build_afl_file_path.is_file(){
            println!("{} build failed.", test);
            failed_count += 1;
        } else {
            println!("{} build success", test);
            success_count += 1;
        }
        let replay = test.clone().replace("test", "replay");
        let build_replay_file_path = target_path.join(&replay);
        if !build_replay_file_path.is_file() {
            println!("{} build failed", replay);
            failed_count += 1;
        } else {
            println!("{} build success", replay);
            success_count += 1;
        }
    }
    // if flag {
    //     println!("check build success");
    // }else {
    //     exit(-1);
    // }
    println!("{} build success, and {} build failed", success_count, failed_count);
}

fn fuzz_it(crate_name: &String, tests: &Vec<String>) {
    println!("start fuzz:");
    let mut test_dir = env::current_dir().unwrap().into_os_string().into_string().unwrap();
    test_dir.push_str("/fuzz_target/");
    test_dir.push_str(crate_name);
    match env::var("Strategy"){
        Ok(value) => {
            test_dir.push_str(("_".to_owned() + &value).as_str());
        },
        Err(_) => {},
    };
    test_dir.push_str("_fuzz");
    let test_path = PathBuf::from(test_dir);
    let test_path = test_path.join(MULTIPLE_TARGETS);
    let target_path = test_path.clone().join("target").join(EDITION);
    let output_path = test_path.clone().join("out");
    let exit_time_path = test_path.join(EXIT_TIME_DIR);
    ensure_empty_dir(&exit_time_path);

    let mut threads = Vec::new();
    let val = Arc::new(AtomicUsize::new(0));


    let mut minute_count = 0;
    let statistic_file_path = test_path.join(STATISTIC_OUTPUT_FILE);
    if statistic_file_path.is_file() {
        fs::remove_file(&statistic_file_path).unwrap();
    }
    if statistic_file_path.is_dir() {
        fs::remove_dir_all(&statistic_file_path).unwrap();
    }
    let mut statisticfile = fs::File::create(&statistic_file_path).unwrap();
    let title = "time\tcrashes\ttargets\tdetails\n";
    statisticfile.write_all(title.as_bytes()).unwrap_or_else(
        |_| {println!("write file {:?} failed.", statistic_file_path); exit(-1);}
    );
    
    // 设置一次spawn fuzz数量为20
    let mut start_fuzz_index = 0;
    let mut end_fuzz_index = 20;
    match env::var("fuzz_num"){
        Ok(value) => {
            match value.as_str(){
                "ALL" => end_fuzz_index = tests.len(),
                _ => end_fuzz_index = value.parse::<usize>().unwrap(),
            }
        },
        Err(_) => {},
    };
    if end_fuzz_index > tests.len(){
        end_fuzz_index = tests.len();
    }

    let mut last_exit_threads_number = 0;
    let mut running_number = 0;
    let mut fuzz_flag = true;
    loop {
        if fuzz_flag {
            running_number += end_fuzz_index - start_fuzz_index;
            for i in start_fuzz_index..end_fuzz_index {
                let test = &tests[i];
                let afl_target_path = target_path.clone().join(test);
                let afl_output_dir = output_path.clone().join(test);
                if afl_output_dir.is_file() {
                    fs::remove_file(&afl_output_dir).unwrap();
                }
        
                let test_path_copy = test_path.clone();
        
                let mut afl_dir_name = test.clone();
                afl_dir_name.push_str("_cmin");
                let mut test_dir2 = env::current_dir().unwrap().into_os_string().into_string().unwrap();
                test_dir2.push_str("/fuzz_target/");
                test_dir2.push_str(crate_name);
                match env::var("Strategy"){
                    Ok(value) => {
                        test_dir2.push_str(("_".to_owned() + &value).as_str());
                    },
                    Err(_) => {},
                };
                test_dir2.push_str("_fuzz");
                let afl_input_path = PathBuf::from(&PathBuf::from(test_dir2)).join(AFL_INPUT_DIR).join(afl_dir_name);
                //println!("fuzz {:?}", afl_target_path);
                //let args = vec!["afl", "fuzz", "-i", AFL_INPUT_DIR, "-o", afl_output_dir.to_str().unwrap(), afl_target_path.to_str().unwrap()];
                //let child = Command::new("cargo").args(args).current_dir(test_path_copy.as_os_str()).stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap();
                let exit_time_file_path = exit_time_path.join(test);
        
                let val_copy = val.clone();
                let handle = thread::spawn(move ||{
                    println!("FUZZING NO.{} --- {:?}", i, afl_target_path);
                    let start = Instant::now();
                    let args = vec!["afl", "fuzz", "-i", afl_input_path.to_str().unwrap(), "-o", afl_output_dir.to_str().unwrap(), afl_target_path.to_str().unwrap()];
                    //println!("{:?}", args);
                    let mut fuzz_command = Command::new("cargo");
                    let exit_status = fuzz_command.args(&args).current_dir(test_path_copy.as_os_str()).env("AFL_EXIT_WHEN_DONE", "1").env("AFL_NO_AFFINITY", "1").stdout(Stdio::null()).status().unwrap();
                    //println!("{:?}", args);
                    println!("FUZZ NO.{} EXITED --- {:?} {:?}", i, afl_target_path, exit_status);
                    println!("FUZZ COMMAND: {:?}\n", fuzz_command);
                    let cost_time = start.elapsed().as_secs();
        
                    //计算退出的数量
                    val_copy.fetch_add(1, Ordering::SeqCst);
        
                    if exit_time_file_path.is_file() {
                        fs::remove_file(&exit_time_file_path).unwrap();
                    }
                    if exit_time_file_path.is_dir() {
                        fs::remove_dir_all(&exit_time_file_path).unwrap();
                    }
                    if exit_status.success() {
                        let mut exit_time_file = fs::File::create(&exit_time_file_path).unwrap();
                        let content = format!("{}", cost_time);
                        exit_time_file.write_all(content.as_bytes()).unwrap_or_else(
                            |_| {println!("write file {:?} failed.", exit_time_file_path); exit(-1);}
                        );
                    }
                });
        
                threads.push(handle);
            }
        }
        // 一分钟更新一次：
        thread::sleep(Duration::from_secs(10));
        minute_count = minute_count + 10;
        println!("----------------------- FUZZ INFO -----------------------");
        println!("FUZZING HAS RUN {} SECONDS.", minute_count);
        output_statistics_to_files(crate_name, minute_count);
        let total_exit_threads_number = val.as_ref().load(Ordering::SeqCst);
        let exit_threads_number = total_exit_threads_number - last_exit_threads_number;
        last_exit_threads_number = total_exit_threads_number;
        println!("EXIT IN PERIOD:\t{}", exit_threads_number);
        if exit_threads_number > 0 {
            running_number -= exit_threads_number;
            start_fuzz_index = end_fuzz_index;
            end_fuzz_index += exit_threads_number;
            if end_fuzz_index > tests.len() {
                end_fuzz_index = tests.len();
            }
            if start_fuzz_index < end_fuzz_index {
                fuzz_flag = true;
            } else {
                fuzz_flag = false;
            }
        } else {
            fuzz_flag = false;
        }
        println!("EXITED THREADS:\t{}\nRUNNING THREADS:{}\nWAITING THREADS:{}", total_exit_threads_number, running_number, tests.len() - total_exit_threads_number - running_number);
        println!("----------------------- FUZZ INFO -----------------------");

        if total_exit_threads_number == tests.len() {
            break;
        }
    }

    println!("Fuzzing totally runs {} minutes. All fuzzing thread finished", minute_count);

    //确保所有的线程都已经退出
    for handle in threads {
        handle.join().unwrap();
    }
}

fn find_crash(crate_name: &String) -> Vec<PathBuf>{
    let mut test_dir = env::current_dir().unwrap().into_os_string().into_string().unwrap();
    // let mut crash_list = Vec::new();
    test_dir.push_str("/fuzz_target/");
    test_dir.push_str(crate_name);
    match env::var("Strategy"){
        Ok(value) => {
            test_dir.push_str(("_".to_owned() + &value).as_str());
        },
        Err(_) => {},
    };
    test_dir.push_str("_fuzz");
    let test_path = PathBuf::from(test_dir);
    let test_path = test_path.join(MULTIPLE_TARGETS);
    let afl_output_path = test_path.join(AFL_OUTPUT_DIR);
    let test_output_paths = check_maybe_empty_directory(&afl_output_path);
    let mut all_crash_files = Vec::new();
    // for test_output_path in &test_output_paths {
    let mut crash_list = vec![0;test_output_paths.len()];
    for i in 0..test_output_paths.len(){
        let test_output_path = &test_output_paths[i];
        let crash_output_path = test_output_path.clone().join(CRASH_DIR);
        // println!("crash_output_path {:?}", crash_output_path);
        let crash_files = check_maybe_empty_directory(&crash_output_path);
        for crash_file in crash_files {
            // println!("crash file {:?}", crash_file);
            let filename = crash_file.to_str().unwrap();
            if !filename.contains("README.txt") {
                crash_list[i] += 1;
                // println!("crash file found for ");
                all_crash_files.push(crash_file);
            }
        }
        if crash_list[i] > 0 {
            println!("Crash for {:?}: {}", test_output_path, crash_list[i]);
        }
    }
    println!("TOTAL CRASH:\t{}", all_crash_files.len());
    all_crash_files
}

fn print_crashes(crate_name: &String) {
    let all_crash_files = find_crash(crate_name);
    if all_crash_files.len() == 0 {
        println!("Find no crash files");
        exit(-1);
    }else {
        for crash in &all_crash_files {
            println!("crash path: {:?}", crash);
        }
    }
    println!("total crashes: {}", all_crash_files.len());
}

fn ensure_empty_dir(dir: &PathBuf) {
    if dir.is_dir() {
        fs::remove_dir_all(dir).unwrap();
    }else if dir.is_file() {
        fs::remove_file(dir).unwrap();
    }
    fs::create_dir_all(dir).unwrap();
}

fn ensure_dir(dir: &PathBuf) {
    if dir.is_file() {
        fs::remove_file(dir).unwrap();
    }
    if !dir.is_dir() {
        fs::create_dir_all(dir).unwrap();
    }
}

fn tmin(crate_name: &String) {
    let all_crash_files = find_crash(crate_name);
    let mut test_dir = env::current_dir().unwrap().into_os_string().into_string().unwrap();
    test_dir.push_str("/fuzz_target/");
    test_dir.push_str(crate_name);
    match env::var("Strategy"){
        Ok(value) => {
            test_dir.push_str(("_".to_owned() + &value).as_str());
        },
        Err(_) => {},
    };
    test_dir.push_str("_fuzz");
    let test_path = PathBuf::from(test_dir);
    let test_path = test_path.join(MULTIPLE_TARGETS);
    let tmin_output_path = test_path.join(TMIN_OUTPUT_DIR);
    ensure_empty_dir(&tmin_output_path);
    if all_crash_files.len() == 0 {
        println!("No crash files.");
        exit(-1);
    }
    println!("total crashes = {}", all_crash_files.len());

    let mut crash_counts = HashMap::new();
    for crash in &all_crash_files {
        let crash_file_name = crash.to_str().unwrap();
        println!("crash_file_name = {}", crash_file_name);
        let file_name_split:Vec<&str> = crash_file_name.split("/").collect();
        let file_name_split_len = file_name_split.len();
        if file_name_split_len < 4 {
            println!("Invalid crash file name");
            exit(-1);
        }
        let test_crate_name = file_name_split[file_name_split_len-4];
        let test_tmin_output_path = tmin_output_path.clone().join(test_crate_name);
        ensure_dir(&test_tmin_output_path);
        let crash_count = if crash_counts.contains_key(test_crate_name) {
            let current_count = *(crash_counts.get(test_crate_name).unwrap()) + 1;
            crash_counts.insert(test_crate_name, current_count);
            current_count 
        }else {
            crash_counts.insert(test_crate_name, 1);
            1
        };
        let target_path = test_path.clone().join("target").join(EDITION).join(test_crate_name);
        let target_file_name = target_path.to_str().unwrap();
        let tmin_output_file = test_tmin_output_path.join(crash_count.to_string());
        let tmin_output_filename = tmin_output_file.to_str().unwrap();
        let tmin_input_filename = crash.to_str().unwrap();
        let args = vec!["afl", "tmin", "-i", tmin_input_filename, "-o", tmin_output_filename, target_file_name];
        //println!("{:?}", args);
        let _ = Command::new("cargo").args(args).stdout(Stdio::null()).output().unwrap();
        //print_output(output);
    }
}

fn cmin(crate_name: &String) {
    let mut test_dir = env::current_dir().unwrap().into_os_string().into_string().unwrap();
    test_dir.push_str("/fuzz_target/");
    test_dir.push_str(crate_name);
    match env::var("Strategy"){
        Ok(value) => {
            test_dir.push_str(("_".to_owned() + &value).as_str());
        },
        Err(_) => {},
    };
    test_dir.push_str("_fuzz");
    let test_path = PathBuf::from(test_dir);
    let test_path = test_path.join(MULTIPLE_TARGETS);
    let cmin_output_path = test_path.clone().join(CMIN_OUTPUT_DIR);
    //如果有tmin的output，首先去找tmin的output
    let tmin_output_dir = test_path.clone().join(TMIN_OUTPUT_DIR);
    if tmin_output_dir.is_dir() {
        let tmin_directories = check_maybe_empty_directory(&tmin_output_dir);
        if tmin_directories.len() > 0 {
            ensure_empty_dir(&cmin_output_path);
            for tmin_directory in tmin_directories {
                let tmin_directory_name = tmin_directory.to_str().unwrap();
                let tmin_directory_name_split: Vec<&str> = tmin_directory_name.split("/").collect();
                let test_case_name = tmin_directory_name_split.last().unwrap();
                //println!("{}", tmin_directory_name);
                execute_cmin(tmin_directory_name, test_case_name, &cmin_output_path, &test_path)
                //print_output(output);
            }
            return;
        }
    }

    //如果没能找到tmin的结果，直接去找crash dir
    let afl_output_path = test_path.join(AFL_OUTPUT_DIR);
    let test_output_paths = check_maybe_empty_directory(&afl_output_path);

    let mut nonempty_crash_dir = Vec::new();

    for test_output_path in &test_output_paths {
        let crash_output_path = test_output_path.clone().join(CRASH_DIR);
        let crash_files = check_maybe_empty_directory(&crash_output_path);
        if crash_files.len() > 0 {
            //如果这个crash目录非空，那么就需要对这个目录运行cmin
            nonempty_crash_dir.push(crash_output_path);
        }
    }

    if nonempty_crash_dir.len() <= 0 {
        println!("no crash file found.");
        exit(-1);
    }

    ensure_empty_dir(&cmin_output_path);

    for crash_dir in nonempty_crash_dir {
        let crash_dir_name = crash_dir.to_str().unwrap();
        //println!("{}", crash_dir_name);
        clean_crash_dir(&crash_dir);
        let crash_dir_name_split:Vec<&str> = crash_dir_name.split("/").collect();
        let crash_dir_name_split_len = crash_dir_name_split.len();
        if crash_dir_name_split_len < 2 {
            println!("Invalid crash dir name");
            exit(-1);
        }
        let test_case_name = crash_dir_name_split[crash_dir_name_split_len - 2];
        execute_cmin(crash_dir_name, test_case_name, &cmin_output_path, &test_path);
        //print_output(output);
    }
}

fn execute_cmin(crash_dir_name: &str, test_case_name:&str, cmin_output_path: &PathBuf, test_path: &PathBuf) {
    println!("{}", test_case_name);
    let test_cmin_output_path = cmin_output_path.clone().join(test_case_name);
    let cmin_output_pathname = test_cmin_output_path.to_str().unwrap();

    let target_path = test_path.clone().join("target").join(EDITION).join(test_case_name);
    let target_name = target_path.to_str().unwrap();

    //println!("crash dir name = {}", crash_dir_name);
    //println!("test case name = {}", test_case_name);
    //println!("cmin output path = {:?}", cmin_output_pathname);
    //println!("test path = {:?}", test_path);
    //println!("target name = {:?}", target_name);

    //add -C option to only apply to crash inputs
    let args = vec!["afl", "cmin", "-C", "-i", crash_dir_name, "-o", cmin_output_pathname, target_name];
    let exit_status = Command::new("cargo").args(args).status().unwrap();
    println!("{:?}", exit_status);
}

//去掉crash dir中的无效文件，比如readme，防止cmin产生不必要的路径
fn clean_crash_dir(crash_dir: &PathBuf) {
    let crash_files = check_maybe_empty_directory(&crash_dir);
    for crash_file in crash_files {
        let crash_filename = crash_file.to_str().unwrap();
        if crash_filename.contains("README.txt") {
            fs::remove_file(crash_file).unwrap();
        }
    }
}

//确认哪些才是真的crash，有些crash可能没法replay
fn replay_crashes(crate_name: &String){
    println!("replay all crashes");
    let mut test_dir = env::current_dir().unwrap().into_os_string().into_string().unwrap();
    test_dir.push_str("/fuzz_target/");
    test_dir.push_str(crate_name);
    match env::var("Strategy"){
        Ok(value) => {
            test_dir.push_str(("_".to_owned() + &value).as_str());
        },
        Err(_) => {},
    };
    test_dir.push_str("_fuzz");
    let test_path = PathBuf::from(test_dir);
    let test_path = test_path.join(MULTIPLE_TARGETS);
    let target_path= test_path.join("target").join(EDITION);
    //如果有cmin的结果的话,那么直接去找cmin的结果
    let cmin_path = test_path.join(CMIN_OUTPUT_DIR);
    if cmin_path.is_dir() {
        let cmin_directories = check_maybe_empty_directory(&cmin_path);
        if cmin_directories.len() > 0 {
            for cmin_directory in cmin_directories {
                if !cmin_directory.is_dir() {
                    continue;
                }
                let crash_files = check_maybe_empty_directory(&cmin_directory);
                if crash_files.len() <= 0 {
                    continue;
                }
                let test_name = last_file_name(&cmin_directory);
                let replay_name = test_name.replace("test", "replay");
                let replay_path = target_path.join(replay_name);
                let replay_file_name = replay_path.to_str().unwrap();
                for crash_file in crash_files {
                    let crash_file_name = crash_file.to_str().unwrap();
                    let output = Command::new(replay_file_name).arg(crash_file_name).output().unwrap();
                    let mut command = replay_file_name.to_string();
                    command.push(' ');
                    command.push_str(crash_file_name);
                    print_output(output);
                    println!("{}", command);
                    //print_output(output);
                }
            }
            return;
        }
    }

    println!("No cmin output files. Use raw crash files");
    //首先尝试直接对原始的结果进行replay
    let mut output_vec = Vec::new();
    let mut command_vec = Vec::new();
    let crash_files = find_crash(crate_name);
    for crash_file in crash_files {
        let crash_file_name = crash_file.to_str().unwrap();
        //println!("{}", crash_file_name);
        //找到replay_file
        let crash_file_name_split:Vec<&str> = crash_file_name.split("/").collect();
        let crash_file_name_split_len = crash_file_name_split.len();
        if crash_file_name_split_len < 4 {
            println!("Invalid crash file name. {}", crash_file_name);
            exit(-1);
        }
        let test_case_name = crash_file_name_split[crash_file_name_split_len - 4];
        //println!("{}", test_case_name);
        let replay_case_name = test_case_name.replace("test", "replay");
        let replay_file_path = target_path.join(replay_case_name);
        let replay_file_name = replay_file_path.to_str().unwrap();
        if !replay_file_path.is_file() {
            println!("Replay file not exist. {}", replay_file_name);
            exit(-1);
        }
        //println!("{}", replay_file_name);
        let output = Command::new(replay_file_name).arg(crash_file_name).output().unwrap();
        let mut command = replay_file_name.to_string();
        command.push(' ');
        command.push_str(crash_file_name);
        println!("{}", command);
        output_vec.push(output.clone());
        command_vec.push(replay_file_name.to_owned() + " " + crash_file_name);
        print_output(output);
    }
    summary_output(&output_vec, &command_vec);
}

pub fn output_statistics(crate_name: &String) {
    //crate_name
    println!("crate name: {}", crate_name);
    //fuzz driver
    let fuzz_drivers = check_pre_condition(crate_name);
    let fuzz_drivers_number = fuzz_drivers.len();
    println!("fuzz drivers: {}", fuzz_drivers_number);
    //total crashes
    let all_crash_files = find_crash(crate_name);
    let crash_number = all_crash_files.len();
    println!("crashes: {}", crash_number);
    //crashes after cmin
    let mut test_dir = env::current_dir().unwrap().into_os_string().into_string().unwrap();
    test_dir.push_str("/fuzz_target/");
    test_dir.push_str(crate_name);
    match env::var("Strategy"){
        Ok(value) => {
            test_dir.push_str(("_".to_owned() + &value).as_str());
        },
        Err(_) => {},
    };
    test_dir.push_str("_fuzz");
    let test_path = PathBuf::from(test_dir);
    let test_path = test_path.join(MULTIPLE_TARGETS);
    
    let cmin_path = test_path.join(CMIN_OUTPUT_DIR);
    if cmin_path.is_dir() {
        let cmin_directories = check_maybe_empty_directory(&cmin_path);
        let find_crash_target_number = cmin_directories.len();
        println!("cmin result(after tmin,cmin):");
        let mut total_cmin_crashes = 0;
        let mut every_target_crashes = String::new();
        if find_crash_target_number > 0 {
            for cmin_directory in &cmin_directories {
                if !cmin_directory.is_dir() {
                    continue;
                }
                let crash_files = check_maybe_empty_directory(&cmin_directory);
                let crash_files_number = crash_files.len();
                every_target_crashes.push_str(format!("\t{} : {}\n", last_file_name(cmin_directory), crash_files_number).as_str());
                total_cmin_crashes = total_cmin_crashes + crash_files_number;
                if crash_files_number <= 0 {
                    continue;
                }
            }
        }
        println!("\tfind crash targets: {}", find_crash_target_number);
        println!("\ttotal crashes after cmin: {}", total_cmin_crashes);
        println!("cmin result detailes: ");
        println!("{}", every_target_crashes);
    }else {
        println!("No cmin output");
    }

    //exit time and average run time
    let exit_path = test_path.join(EXIT_TIME_DIR);
    let exit_targets = check_maybe_empty_directory(&exit_path);
    let finished_targets_number = exit_targets.len();
    println!("targets finished : {} ", finished_targets_number);

    let invalid_targets_number = if INVALID_TARGET_NUMBER.contains_key(crate_name.as_str()) {
        INVALID_TARGET_NUMBER.get(crate_name.as_str()).unwrap().clone()
    } else {
        0
    };

    println!("invalid targets: {}", invalid_targets_number);

    let mut run_time:Vec<u64> = Vec::new();
    let run_to_end_targets = fuzz_drivers_number - finished_targets_number - invalid_targets_number;
    for _ in 0..run_to_end_targets {
        run_time.push(86400);
    }

    for exit_target in &exit_targets {
        //println!("{:?}", exit_target);
        let content = fs::read_to_string(exit_target).expect("read exit file error");
        let mut  exit_time = content.parse::<u64>().unwrap();
        if exit_time > 86400 {
            exit_time = 86400;
        }
        //println!("{}", exit_time);
        run_time.push(exit_time);
    }

    println!("valid targets: {}", run_time.len());
    //println!("{:?}", run_time);

    let run_time_sum:u64 = run_time.iter().sum();
    println!("average run time: {}", (run_time_sum as f64)/ (((run_time.len()*3600) as f64)));
}

pub fn output_statistics_to_files(crate_name: &String, fuzz_time: usize) {
    let mut test_dir = env::current_dir().unwrap().into_os_string().into_string().unwrap();
    test_dir.push_str("/fuzz_target/");
    test_dir.push_str(crate_name);
    match env::var("Strategy"){
        Ok(value) => {
            test_dir.push_str(("_".to_owned() + &value).as_str());
        },
        Err(_) => {},
    };
    test_dir.push_str("_fuzz");
    let test_path = PathBuf::from(test_dir);
    let test_path = test_path.join(MULTIPLE_TARGETS);
    let statistic_file_path = test_path.join(STATISTIC_OUTPUT_FILE);
    if !statistic_file_path.is_file() {
        fs::File::create(&statistic_file_path).unwrap();
    }

    let all_crash_files = find_crash(crate_name);
    let total_crash_number = all_crash_files.len();

    let mut crash_counts = HashMap::new();
    for crash in &all_crash_files {
        let crash_file_name = crash.to_str().unwrap();
        let file_name_split:Vec<&str> = crash_file_name.split("/").collect();
        let file_name_split_len = file_name_split.len();
        if file_name_split_len < 4 {
            println!("Invalid crash file name");
            exit(-1);
        }
        let test_crate_name = file_name_split[file_name_split_len-4];
        if crash_counts.contains_key(test_crate_name) {
            let current_count = *(crash_counts.get(test_crate_name).unwrap()) + 1;
            crash_counts.insert(test_crate_name, current_count);
        }else {
            crash_counts.insert(test_crate_name, 1);
        };
    }

    let crash_targets_number = crash_counts.len();
    let mut detail = "[".to_string();
    //detail:即每个target对应的crash的数量
    for (crate_name, crash_number) in &crash_counts {
        detail.push_str(*crate_name);
        detail.push_str(" : ");
        detail.push_str(format!("{}", crash_number).as_str());
        detail.push_str(" ,");
    }
    detail.push_str("]");

    let content = format!("{}\t{}\t{}\t{}\t\n", fuzz_time, total_crash_number, crash_targets_number, detail);

    let mut file = OpenOptions::new().append(true).open(&statistic_file_path).unwrap_or_else(
        |_| {println!("can't open file {:?}.", statistic_file_path); exit(-1);}
    );

    file.write_all(content.as_bytes()).unwrap_or_else(
        |_| {println!("write file {:?} failed.", statistic_file_path); exit(-1);}
    );
}

fn showmap(crate_name: &String) {
    let mut test_dir = env::current_dir().unwrap().into_os_string().into_string().unwrap();
    test_dir.push_str("/fuzz_target/");
    test_dir.push_str(crate_name);
    match env::var("Strategy"){
        Ok(value) => {
            test_dir.push_str(("_".to_owned() + &value).as_str());
        },
        Err(_) => {},
    };
    test_dir.push_str("_fuzz");
    let test_path = PathBuf::from(test_dir);
    let test_path = test_path.join(MULTIPLE_TARGETS);
    let showmap_path = test_path.join(SHOWMAP_DIR);
    ensure_empty_dir(&showmap_path);
    let tests = check_pre_condition(crate_name);
    for test in &tests {
        let out_dir = test_path.join(AFL_OUTPUT_DIR).join(test).join("default");
        //println!("{:?}",out_dir.as_os_str());
        if !out_dir.is_dir() {
            println!("{} has no output dir", test);
            continue;
        }
        let target_path = test_path.join("target").join("debug").join(test);
        let showmap_file_path = showmap_path.join(test);
        let output = Command::new("cargo").arg("afl").arg("showmap").arg("-C").arg("-e")
            .arg("-i").arg(out_dir.as_os_str()).arg("-o").arg(showmap_file_path.as_os_str()).arg("--").arg(target_path.as_os_str())
            .current_dir(test_path.as_os_str()).output().unwrap();
        let stdout = &output.stdout;
        let stdout_content = str::from_utf8(stdout).unwrap();
        //println!("{}", stdout_content);
        let stdout_content_lines: Vec<&str> = stdout_content.split("\n").collect();
        let stdout_lines_len = stdout_content_lines.len();
        if stdout_lines_len > 1 {
            let last_line = stdout_content_lines[stdout_lines_len - 2];
            println!("{}", last_line);
        }
    }
}

fn init_afl_input(crate_name: &String) {
    let mut test_dir = env::current_dir().unwrap().into_os_string().into_string().unwrap();
    test_dir.push_str("/fuzz_target/");
    test_dir.push_str(crate_name);
    match env::var("Strategy"){
        Ok(value) => {
            test_dir.push_str(("_".to_owned() + &value).as_str());
        },
        Err(_) => {},
    };
    test_dir.push_str("_fuzz");
    let test_path = PathBuf::from(test_dir);
    let afl_init_path = test_path.join(AFL_INPUT_DIR);
    let test_path = test_path.join(MULTIPLE_TARGETS);
    let afl_directory_paths = check_no_empty_directory(&afl_init_path);    
    let mut afl_files = Vec::new();

    for afl_path in &afl_directory_paths {
        if afl_path.is_file() {
            afl_files.push(afl_path.clone());
        }
    }

    let tests = check_pre_condition(crate_name);
    for test in &tests {
        let replay = test.replace("test", "replay");
        let this_afl_init_path = afl_init_path.join(test);
        ensure_empty_dir(&this_afl_init_path);
        let replay_target_path = test_path.join("target").join("debug").join(&replay);
        let test_target_path = test_path.join("target").join("debug").join(test);
        let mut has_init_file_flag = false;

        // println!("checking {:?}", replay_target_path);
        let mut check_flag = false;
        for afl_file in &afl_files {
            if replay_target_path.is_file(){
                let exit_status = Command::new(replay_target_path.as_os_str()).arg(afl_file.as_os_str()).stdout(Stdio::null()).stderr(Stdio::null()).status().unwrap_or_else(
                    |e| {println!("check error for {:?}, {:#?}\nError:{:#?}", replay_target_path, afl_file, e); exit(-1);}
                );
                if exit_status.success() {
                    has_init_file_flag = true;
                    Command::new("cp").arg(afl_file.as_os_str()).arg(this_afl_init_path.as_os_str()).status().unwrap();
                }
                check_flag = true;
            }
        }
        if !check_flag {
            continue;
        }
        
        //tmin:慢
        //let mut tmin_name = test.clone();
        //tmin_name.push_str("_tmin");
        //let this_tmin_path = afl_init_path.join(&tmin_name);
        //ensure_empty_dir(&this_tmin_path);
        //let all_raw_afl_files = check_maybe_empty_directory(&this_afl_init_path);
        //for raw_afl_file in &all_raw_afl_files {
        //    let filename = last_file_name(raw_afl_file);
        //    let output_file_path = this_tmin_path.join(filename);
        //    let args = vec!["afl", "tmin", "-i", raw_afl_file.to_str().unwrap(), "-o", output_file_path.to_str().unwrap(), "--", test_target_path.to_str().unwrap()];
        //    let _ = Command::new("cargo").args(&args).stdout(Stdio::null()).stderr(Stdio::null()).status().unwrap();
        //}

        if !has_init_file_flag {
            println!("There's no afl input for {:?}", test);
        }else {
            println!("afl cmin for {:?}", test);
            let mut cmin_name = test.clone();
            cmin_name.push_str("_cmin");
            let this_cmin_path = afl_init_path.join(&cmin_name);
            ensure_empty_dir(&this_cmin_path);
            let cmin_args = vec!["afl", "cmin", "-i", this_afl_init_path.to_str().unwrap(), "-o", this_cmin_path.to_str().unwrap(), "--", test_target_path.to_str().unwrap()];
            // println!("afl cmin: {:#?}", cmin_args);
            let _ = Command::new("cargo").args(&cmin_args).stdout(Stdio::null()).status().unwrap();
        }
    }
}


fn main() {
    let args: Vec<String> = env::args().collect();
    let user_options = UserOptions::new_from_cli(args);
    println!("{:?}", user_options);
    do_work(&user_options);
}
