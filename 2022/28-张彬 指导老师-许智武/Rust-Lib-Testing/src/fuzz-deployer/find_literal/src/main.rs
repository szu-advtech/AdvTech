use std::env;
extern crate regex;
extern crate rand;
use regex::Regex;
use std::path::PathBuf;
use std::process;
use std::fs;
use std::io::{self, Write};
use std::collections::{HashMap, BinaryHeap};
use rand::prelude::*;
use rand::distributions::weighted::alias_method::WeightedIndex;

#[derive(Debug, Clone)]
pub struct UserOptions {
    output_option: Option<PathBuf>,
    input_option: Option<PathBuf>,
    suggest_length: Option<usize>,
    afl_initail_number: Option<usize>,
}

impl UserOptions {
    pub fn new() -> Self {
        UserOptions {
            output_option: None,
            input_option: None,
            suggest_length: None,
            afl_initail_number: None,
        }
    }

    pub fn new_from_cli(args: &Vec<String>) -> Self {
        let mut user_options = UserOptions::new();
        user_options.extract_options(args);
        user_options.set_default_option();
        user_options
    }

    pub fn extract_options(&mut self, args: &Vec<String>){
        let output_option = Regex::new("(-o|--output)").unwrap();
        let input_option = Regex::new("(-i|--input)").unwrap();
        let length_option = Regex::new("(-l|--length)").unwrap();
        let number_option = Regex::new("(-n|--number)").unwrap();
        let helper_option = Regex::new("(-h|--help)").unwrap();
        let mut args_iter = args.iter();
        let _ = args_iter.next(); // 把程序的名字跳过
        while let Some(s) = args_iter.next() {
            //extract output option
            if output_option.is_match(s.as_str()) {
                if let Some(output_dir) = args_iter.next() {
                    let pathbuf = PathBuf::from(output_dir);  
                    self.output_option = Some(pathbuf); 
                    continue;
                }else {
                    println!("Invalid options. Expect output directory after -o/--output");
                    process::exit(-1);
                }
            }
            //extract input option
            if input_option.is_match(s.as_str()) {
                if let Some(input_file) = args_iter.next() {
                    let pathbuf = PathBuf::from(input_file);
                    self.input_option = Some(pathbuf);
                    continue;
                }else {
                    println!("Invalid options. Expect input directory after -i/--input");
                    process::exit(-2);
                }
            }
            if length_option.is_match(s.as_str()) {
                if let Some(length) = args_iter.next() {
                    let length = length.parse::<usize>();
                    if length.is_err() {
                        println!("Invalid options. Expect valid number after -l/--length");
                        process::exit(-2);
                    }
                    let length = length.unwrap();
                    self.suggest_length = Some(length);
                    continue;
                }else {
                    println!("Invalid options. Expect number after -l/--length");
                    process::exit(-3);
                }
            }
            if number_option.is_match(s.as_str()) {
                if let Some(number) = args_iter.next() {
                    let number = number.parse::<usize>();
                    if number.is_err() {
                        println!("Invalid options. Expect valid number after -n/--number");
                        process::exit(-4);
                    }
                    let number = number.unwrap();
                    self.afl_initail_number = Some(number);
                    continue;
                } else {
                    println!("Invalid options. Expect valid number after -n/--number");
                    process::exit(-5);
                }
            }
            if helper_option.is_match(s.as_str()) {
                if let Some(_) = args_iter.next() {
                    println!("Invalid options.\n");
                    println!("{}", help_message());
                    process::exit(-12);
                }else {
                    println!("{}", help_message());
                    process::exit(0);
                }
            }

            println!("Not found param: {}", s.as_str());
            process::exit(-13);
        }
    }

    pub fn set_default_option(&mut self) {
        if self.output_option.is_none() {
            self.output_option = Some(PathBuf::from("."));
        }
        if self.input_option.is_none() {
            self.input_option = Some(PathBuf::from("."));
        }
        if self.afl_initail_number.is_none() {
            self.afl_initail_number = Some(20);
        }
    }
}

pub fn help_message() -> &'static str {
"find_literal 0.1.0

USAGE: 
    find_literal (-i input_dir) (-o output_dir) (-l length) (-n number)
    
FLAGS:
    -o,--output     set output directory(default .)
    -i,--input      set input directory/file(default .)
    -l,--length     set sequence length(default random from 1 to 5)
    -n,--number     set sequence number(default 20)
    -h,--help       print help message
"
}

pub fn read_literals(user_options: &UserOptions) -> Vec<String>{
    let input_filename = user_options.input_option.as_ref().unwrap();

    let mut all_rs_files = Vec::new();

    if input_filename.is_dir() {
        walk_dir_recursive(input_filename, &mut all_rs_files);
    }

    if input_filename.is_file() {
        if is_rs_src_file(input_filename) {
            all_rs_files.push(input_filename.clone());
        }
    }
    all_rs_files.sort();
    println!("all_rs_files = {:?}", all_rs_files);

    let mut integer_literals = HashMap::new();
    let mut float_literals = HashMap::new();
    let mut string_literals = HashMap::new();
    let mut char_literals = HashMap::new();
    //slice字面量是否需要？
    //let mut slice_literals = Vec::new();

    for file in &all_rs_files {
        read_file(file, &mut integer_literals, &mut float_literals, &mut string_literals, &mut char_literals);
    }

    let initial_inputs = generate_afl_initial_input(user_options, &integer_literals, &float_literals, &string_literals, &char_literals);
    //println!("{:?}", initial_inputs);
    initial_inputs
}

pub fn walk_dir_recursive(path: &PathBuf, all_files: &mut Vec<PathBuf>) {
    let dir_entry = fs::read_dir(path).unwrap();
    let dir_entries = dir_entry.map(|res| res.map(|e| e.path())).collect::<Result<Vec<_>, io::Error>>().unwrap();
    for dir_path in &dir_entries {
        if dir_path.is_file() {
            if is_rs_src_file(dir_path) {
                all_files.push(dir_path.clone());
            }
        }
        if dir_path.is_dir() {
            walk_dir_recursive(dir_path, all_files);
        }
    }
}

pub fn is_rs_src_file(path: &PathBuf) -> bool {
    let rs_src_file = Regex::new("\\.rs$").unwrap();
    let filename = path.to_str().unwrap();
    if rs_src_file.find(filename).is_some() {
        true
    }else {
        false
    }
}

pub fn read_file(path: &PathBuf, integer_literals: &mut HashMap<String, usize>, float_literals: &mut HashMap<String,usize>,
    string_literals: &mut HashMap<String, usize>, char_literals: &mut HashMap<String,usize>) {
    //println!("filename = {:?}", path.as_os_str());
    let rs_file = fs::read_to_string(path);
    if rs_file.is_err() {
        return;
    }
    let rs_file = rs_file.unwrap();
    let rs_file_lines = rs_file.split("\n");
    let integer_regex = Regex::new("(\\+|-)?(\\d)+(e(\\+|-)?(\\d)+)?").unwrap();
    let float_regex = Regex::new("(\\+|-)?(\\d)+\\.((\\+|-)?(\\d)+(e(\\+|-)?(\\d)+)?)?").unwrap();
    let string_regex = Regex::new("\"(^\"|\\s|\\S)*\"").unwrap();
    let char_regex = Regex::new("\'\\w\'").unwrap();
    for line in rs_file_lines {
        let integer_matches = integer_regex.find_iter(line);
        for integer_match in integer_matches {
            let integer_literal = integer_match.as_str().to_string();
            add_to_hashmap(integer_literal, integer_literals);
        }
        let float_matches = float_regex.find_iter(line);
        for float_match in float_matches {
            let float_literal = float_match.as_str().to_string();
            add_to_hashmap(float_literal, float_literals);
        }
        let string_matches = string_regex.find_iter(line);
        for string_match in string_matches {
            let string_literal= string_match.as_str().to_string().replace("\"", "");
            if string_literal.contains("{}") || string_literal.contains("{:?}") || string_literal.contains("{:#?}}") {
                continue;
            }
            add_to_hashmap(string_literal, string_literals)
        }
        let char_matches = char_regex.find_iter(line);
        for char_match in char_matches {
            let char_literal = char_match.as_str().to_string().replace("\'", "");
            add_to_hashmap(char_literal, char_literals);
        }
    }
}

pub fn add_to_hashmap(key: String, map: &mut HashMap<String, usize>) {
    let count = if map.contains_key(&key) {
        map.get(&key).unwrap() + 1
    }else {
        1
    };
    map.insert(key, count);
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
struct LiteralCount {
    count: usize,
    literal: String,
}

pub fn generate_afl_initial_input(user_options: &UserOptions,integer_literals: &HashMap<String, usize>, float_literals: &HashMap<String,usize>,
        string_literals: &HashMap<String, usize>, char_literals: &HashMap<String,usize>) -> Vec<String> {

    let mut literal_counts = BinaryHeap::new();

    add_literal_counts(&mut literal_counts, integer_literals);
    add_literal_counts(&mut literal_counts, float_literals);
    add_literal_counts(&mut literal_counts, string_literals);
    add_literal_counts(&mut literal_counts, char_literals);

    let literal_vec = literal_counts.iter()
        .map(|literal_count| literal_count.literal.clone())
        .collect::<Vec<String>>();
    let literal_weights = literal_counts.iter()
        .map(|literal_count| literal_count.count)
        .collect::<Vec<usize>>();
    let dist = WeightedIndex::new(literal_weights).unwrap();
    
    let number = user_options.afl_initail_number.unwrap();
    let length  = user_options.suggest_length;
    let literal_number = literal_vec.len();
    
    let mut res = Vec::new();
    if literal_number == 0 {
        println!("no literal has been found in input directory{:?}.", user_options.input_option.as_ref().unwrap());
        process::exit(-6);
    }
    let mut rng = thread_rng();

    let length_weight:Vec<usize> = vec![4,2,1,1];
    let length_dist = WeightedIndex::new(length_weight).unwrap();
    for _ in 0..number {
        let mut one_string = String::new();
        let pick_number = if length.is_none() {
            length_dist.sample(&mut rng) + 1
        }else {
            length.unwrap()
        };

        for _ in 0..pick_number {
            let pick_index = dist.sample(&mut rng);
            one_string.push_str(literal_vec[pick_index].as_str());
        }
        res.push(one_string);
    }
    res
}

fn add_literal_counts(literal_counts: &mut BinaryHeap<LiteralCount>, literals: &HashMap<String, usize>) {
    for (literal, count) in literals {
        let literal_count = LiteralCount {
            count: count.clone(),
            literal: literal.clone(),
        };
        literal_counts.push(literal_count);
    }
}

pub fn write_to_file(user_options: &UserOptions, initial_inputs: Vec<String>) {
    let output_option = user_options.output_option.as_ref().unwrap();
    if !output_option.is_dir() {
        println!("Output option is not a directory. {:#?}", output_option);
        process::exit(-7);
    }
    let output_dir = &output_option.clone().join("afl_init");
    if output_dir.is_dir() {
        fs::remove_dir_all(output_dir).unwrap_or_else(
            |_| {println!("Unable to delete output directory."); process::exit(-8)}
        );
    }
    if output_dir.is_file() {
        fs::remove_file(output_dir).unwrap_or_else(
            |_| {println!("Unable to delete output file."); process::exit(-9);}
        )
    }
    fs::create_dir_all(output_dir).unwrap_or_else(
        |_| {println!("Unable to create output directory {:?}.", output_dir); process::exit(-10);}
    );

    let initial_inputs_num = initial_inputs.len();
    for i in 0..initial_inputs_num {
        let filename = format!("afl_input{}", i);
        let output_file = &output_dir.clone().join(filename);
        let mut output_file = fs::File::create(output_file).unwrap_or_else(
            |_| {println!("Unable to create output file {:?}", output_file); process::exit(-11);}
        );
        let write_content = &initial_inputs[i];
        output_file.write_all(write_content.as_bytes()).unwrap_or_else(
            |_| {println!("write file {:?} failed.", output_file); process::exit(-12);}
        );
    }

    println!("output_dir = {:?}", output_dir);
}

fn main() {
    let args:Vec<String> = env::args().collect();
    let user_options = UserOptions::new_from_cli(&args);
    //println!("{:?}",user_options);

    let initial_inputs= read_literals(&user_options);
    write_to_file(&user_options, initial_inputs);
}
