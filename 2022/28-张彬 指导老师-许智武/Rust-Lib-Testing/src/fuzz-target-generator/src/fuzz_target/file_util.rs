use std::collections::HashMap;
use crate::fuzz_target::api_graph::ApiGraph;
use std::path::PathBuf;
use std::fs;
use std::io::Write;
use std::env;
pub use lazy_static::*;

lazy_static! {
    static ref RANDOM_TEST_FILE_NUMBERS: HashMap<&'static str, usize> = {
        let mut m = HashMap::new();
        m.insert("url", 61);
        m.insert("regex", 67);
        m.insert("time", 118);
        m
    };
}

static _TEST_FILE_DIR: &'static str = "test_files";
static _REPRODUCE_FILE_DIR: &'static str = "replay_files";
static _LIBFUZZER_DIR_NAME: &'static str = "libfuzzer_files";
static MAX_TEST_FILE_NUMBER: usize = 300;
static DEFAULT_RANDOM_FILE_NUMBER: usize = 300;

pub fn get_dir_path(crate_name: &String) -> String {
    let mut crate_test_dir = env::current_dir().unwrap().into_os_string().into_string().unwrap(); // modified by wcventure
    crate_test_dir.push_str("/fuzz_target/");
    let search_method = match env::var("Strategy"){ 
        Ok(value) => {
            "_".to_owned() + &value
        },
        Err(_) => "".to_string()
    };
    crate_test_dir = crate_test_dir + crate_name + &search_method + "_fuzz";
    print!("CRATE TEST DIR: {}\n", crate_test_dir);
    crate_test_dir
}

pub fn get_dot_path(crate_name: &String) -> String {
    let mut crate_dot_path = env::current_dir().unwrap().into_os_string().into_string().unwrap(); // modified by wcventure
    crate_dot_path.push_str("/fuzz_target/");
    let search_method = match env::var("Strategy"){ 
        Ok(value) => {
            "_".to_owned() + &value
        },
        Err(_) => "".to_string()
    };
    crate_dot_path = crate_dot_path + crate_name + &search_method + "_graph" + ".dot";
    print!("CRATE DOT: {}\n", crate_dot_path);
    crate_dot_path
}

pub fn get_api_path(crate_name: &String) -> String {
    let mut crate_api_path = env::current_dir().unwrap().into_os_string().into_string().unwrap(); // modified by wcventure
    crate_api_path.push_str("/fuzz_target/");
    let search_method = match env::var("Strategy"){ 
        Ok(value) => {
            "_".to_owned() + &value
        },
        Err(_) => "".to_string()
    };
    crate_api_path = crate_api_path + crate_name + &search_method + "_api" + ".txt";
    print!("CRATE API: {}\n", crate_api_path);
    crate_api_path
}

pub fn get_dependency_path(crate_name: &String) -> String {
    let mut crate_dependency_path = env::current_dir().unwrap().into_os_string().into_string().unwrap(); // modified by wcventure
    crate_dependency_path.push_str("/fuzz_target/");
    let search_method = match env::var("Strategy"){ 
        Ok(value) => {
            "_".to_owned() + &value
        },
        Err(_) => "".to_string()
    };
    crate_dependency_path = crate_dependency_path + crate_name + &search_method + "_dependency" + ".txt";
    print!("CRATE DEPENDENCY: {}\n", crate_dependency_path);
    crate_dependency_path
}

pub fn get_libfuzzer_dir(crate_name: &String) -> String {
    let mut crate_test_dir = env::current_dir().unwrap().into_os_string().into_string().unwrap(); // modified by wcventure
    crate_test_dir.push_str("/libfuzzer_target/");
    let search_method = match env::var("Strategy"){ 
        Ok(value) => {
            "_".to_owned() + &value
        },
        Err(_) => "".to_string()
    };
    crate_test_dir = crate_test_dir + crate_name + &search_method + "_fuzz";
    print!("LIBFUZZER_FUZZ_TARGET_DIR: {}\n", crate_test_dir);
    crate_test_dir
}

#[derive(Debug, Clone)]
pub struct FileHelper {
    pub crate_name: String,
    pub test_dir: String,
    pub test_files: Vec<String>,
    pub reproduce_files: Vec<String>,
    pub libfuzzer_files: Vec<String>,
}

impl FileHelper {
    pub fn new(api_graph: &ApiGraph, random_strategy: bool) -> Self{
        let mut crate_name = api_graph._crate_name.clone();
        let random_test_dir:String = "_Random".to_owned();
        let test_dir = if !random_strategy {
            get_dir_path(&crate_name)
        } else {
            crate_name.push_str(&random_test_dir);
            get_dir_path(&crate_name)
        };
        let mut sequence_count = 0;
        let mut test_files = Vec::new();
        let mut reproduce_files = Vec::new();
        let mut libfuzzer_files = Vec::new();
        let chosen_sequences = match env::var("GenChoose"){
            Ok(value) => {
                match value.as_str(){
                    "all" => api_graph._all_choose(),
                    "naive" => api_graph._naive_choose_sequence(MAX_TEST_FILE_NUMBER),
                    "random" => api_graph._random_choose(DEFAULT_RANDOM_FILE_NUMBER),
                    "heuristic" => api_graph._heuristic_choose(MAX_TEST_FILE_NUMBER, true),
                    "unsafe" => api_graph._unsafe_choose(),
                    "unsafeHeuristic" => api_graph._unsafe_heuristic_choose(MAX_TEST_FILE_NUMBER, true),
                    "pattern" => api_graph._pattern_choose(MAX_TEST_FILE_NUMBER),
                    "newHeuristic" => api_graph._new_heuristic_choose(MAX_TEST_FILE_NUMBER, true),
                    _ => api_graph._all_choose(),
                }
            },
            Err(_) => api_graph._all_choose(),
        };
        api_graph.create_api_dependency_graph_visualize();
        api_graph.evaluate_sequences();
        // let chosen_sequences = api_graph._naive_choose_sequence(MAX_TEST_FILE_NUMBER);
        // let chosen_sequences = api_graph._random_choose(DEFAULT_RANDOM_FILE_NUMBER);
        // let chosen_sequences = api_graph._all_choose();
        // let chosen_sequences = if !random_strategy {
        //     api_graph._heuristic_choose(MAX_TEST_FILE_NUMBER, true)
        // } else {
        //     let random_size = if RANDOM_TEST_FILE_NUMBERS.contains_key(crate_name.as_str()) {
        //         (RANDOM_TEST_FILE_NUMBERS.get(crate_name.as_str()).unwrap()).clone()
        //     } else {
        //         DEFAULT_RANDOM_FILE_NUMBER
        //     };
        //     api_graph._first_choose(random_size)
        // };
        //println!("chosen sequences number: {}", chosen_sequences.len());

        for sequence in &chosen_sequences{
            if sequence_count >= MAX_TEST_FILE_NUMBER {
                break;
            }
            let test_file = sequence._to_afl_test_file(api_graph, sequence_count);
            test_files.push(test_file);
            let reproduce_file = sequence._to_replay_crash_file(api_graph, sequence_count);
            reproduce_files.push(reproduce_file);
            let libfuzzer_file = sequence._to_libfuzzer_test_file(api_graph, sequence_count);
            libfuzzer_files.push(libfuzzer_file);
            sequence_count = sequence_count + 1;
        }
        FileHelper {
            crate_name,
            test_dir,
            test_files,
            reproduce_files,
            libfuzzer_files,
        }
    }

    pub fn write_files(&self) {
        let test_path = PathBuf::from(&self.test_dir);
        if test_path.is_file() {
            fs::remove_file(&test_path).unwrap();
        }
        let test_file_path = test_path.clone().join(_TEST_FILE_DIR);
        let reproduce_file_path = test_path.clone().join(_REPRODUCE_FILE_DIR);
        if ensure_empty_dir(&test_file_path) & ensure_empty_dir(&reproduce_file_path) {
            write_to_files(&self.crate_name, &test_file_path, &self.test_files, "test");
            //暂时用test file代替一下，后续改成真正的reproduce file
            write_to_files(&self.crate_name, &reproduce_file_path, &self.reproduce_files, "replay");
        }
    }

    pub fn write_libfuzzer_files(&self) {
        let libfuzzer_dir = get_libfuzzer_dir(&self.crate_name);
        let libfuzzer_path = PathBuf::from(libfuzzer_dir);
        if libfuzzer_path.is_file() {
            fs::remove_file(&libfuzzer_path).unwrap();
        }
        let libfuzzer_files_path = libfuzzer_path.join(_LIBFUZZER_DIR_NAME);
        if ensure_empty_dir(&libfuzzer_files_path){
            write_to_files(&self.crate_name, &libfuzzer_files_path, &self.libfuzzer_files, "fuzz_target");
        }
    }
}

fn write_to_files(crate_name: &String, path: &PathBuf, contents: &Vec<String>, prefix: &str) {
    let file_number = contents.len();
    for i in 0..file_number {
        let filename = format!("{}_{}{}.rs",prefix, crate_name, i);
        let full_filename = path.join(filename);
        let mut file = fs::File::create(full_filename).unwrap();
        file.write_all(contents[i].as_bytes()).unwrap();
    }
}

fn ensure_empty_dir(path: &PathBuf) -> bool{
    if path.is_file() {
        fs::remove_file(path).unwrap();
    }
    if path.is_dir() {
        fs::remove_dir_all(path).unwrap();
    }
    match fs::create_dir_all(path) {
        Ok(value) => {true},
        Err(_) => { print!("{:?} can't create\n", path); false}
    }
}

