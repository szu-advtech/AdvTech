use std::env;
use std::ffi::OsString;
use std::process::Command;

const CARGO_LOCK_BUG_DETECTOR_HELP: &str = r#"Detect double-lock&conflict-lock on MIR
Usage:
    cargo rosa [subcommand] [<cargo options>...] [--] [<program/test suite options>...]
Subcommands:
    double-lock              Detect double-lock bugs
    conflict-lock            Detect conflict-lock bugs
    operation-sequence       Find operation sequence based on pattern
Common options:
    -h, --help               Print this message
    -V, --version            Print version info and exit
Other [options] are the same as `cargo check`. Everything after the second "--" verbatim
to the program.
Examples:
    cargo rosa double-lock
    cargo rosa conflict-lock
    cargo rosa operation-sequence
"#;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum BugDetectCommand {
    OperationSequence,
    DoubleLock,
    ConflictLock,
    ApiDependencyGraph,
}

fn show_help() {
    println!("{}", CARGO_LOCK_BUG_DETECTOR_HELP);
}

fn show_version() {
    println!("rosa {}", "1.0.0");
}

fn show_error(msg: String) -> ! {
    eprintln!("fatal error: {}", msg);
    std::process::exit(1)
}

fn cargo() -> Command {
    Command::new(env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo")))
}

// Determines whether a `--flag` is present.
fn has_arg_flag(name: &str) -> bool {
    let mut args = std::env::args().take_while(|val| val != "--");
    args.any(|val| val == name)
}

fn in_cargo_operation_sequence_analyze() {
    let (subcommand, skip) = match std::env::args().nth(2).as_deref() {
        Some("operation-sequence") => (BugDetectCommand::OperationSequence, 3),
        Some("double-lock") => (BugDetectCommand::DoubleLock, 3),
        Some("conflict-lock") => (BugDetectCommand::ConflictLock, 3),
        Some("api-dependency-graph") => (BugDetectCommand::ApiDependencyGraph, 3),
        // Default operation-sequence
        None => (BugDetectCommand::OperationSequence, 2),
        // Invalid command.
        Some(s) => show_error(format!("Unknown command `{}`", s)),
    };
    // Now we run `cargo check $FLAGS $ARGS`, giving the user the
    // change to add additional arguments. `FLAGS` is set to identify
    // this target.  The user gets to control what gets actually passed to operation-sequence detect.
    let mut cmd = cargo();
    cmd.arg("check");
    match subcommand {
        BugDetectCommand::OperationSequence => {
            cmd.env("RUST_DETECTOR_TYPE", "OperationSequenceDetector")
        }
        BugDetectCommand::DoubleLock => {
            cmd.env("RUST_DETECTOR_TYPE", "DoubleLockDetector")
        }
        BugDetectCommand::ConflictLock => {
            cmd.env("RUST_DETECTOR_TYPE", "ConflictLockDetector")
        }
        BugDetectCommand::ApiDependencyGraph => {
            cmd.env("RUST_DETECTOR_TYPE", "ApiDependencyGraphGenerator")
        }
    };
    cmd.env("RUSTC", "rust-operation-sequence-analyzer");
    cmd.env("RUST_BACKTRACE", "full");
    let mut args = std::env::args().skip(skip);
    while let Some(arg) = args.next() {
        if arg == "--" {
            break;
        }
        cmd.arg(arg);
    }
    cmd.env("RUST_DETECTOR_BLACK_LISTS", "cc");
    println!("{:?}", cmd);
    let exit_status = cmd
        .spawn()
        .expect("could not run cargo")
        .wait()
        .expect("failed to wait for cargo?");
    println!("[ROSA] Finished all!");
    if !exit_status.success() {
        std::process::exit(exit_status.code().unwrap_or(-1))
    };
}

fn main() {
    if has_arg_flag("--help") || has_arg_flag("-h") {
        show_help();
        return;
    }
    if has_arg_flag("--version") || has_arg_flag("-V") {
        show_version();
        return;
    }
    if let Some("rosa") = std::env::args().nth(1).as_deref() {
        in_cargo_operation_sequence_analyze();
    }
}
