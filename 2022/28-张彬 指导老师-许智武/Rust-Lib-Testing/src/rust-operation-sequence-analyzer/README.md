# rust-operation-sequence-analyzer

This project is our initial efforts to improve the memory safety bug in Rust ecosystem by statically find operation-sequence on Rust MIR based on our pre-defined pattern.

## Install

Currently supports rustc version: 1.51.0-nightly (c3abcfe8a 2021-01-25)
```
$ https://github.com/SZU-SE/Rust-Lib-Testing.git --recursive 
$ cd rust-operation-sequence-analyzer
$ rustup component add rust-src rustc-dev llvm-tools-preview
$ cargo install --path .
```

## Example

Move to test examples
```
$ cd rust-operation-sequence-analyzer
$ ./run.sh examples/queue
```

Run with cargo subcommands
```
$ cd rust-operation-sequence-analyzer/examples/queue
$ cargo clean
$ cargo rosa
```
You need to run
```
cargo clean
```
before re-detecting.

The example (unsafe queue):
```rust
#[derive(Debug)]
struct Queue<T> {
    qdata: Vec<T>,
}

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

impl<T> Queue<T> {
    fn new() -> Self {
        Queue { qdata: Vec::new() }
    }

    fn push(&mut self, item: T) {
        self.qdata.push(item);
    }

    fn pop(&self) -> Option<T> {
        let l = self.qdata.len();
        if l > 0 {
            let pptr = &self.qdata as *const Vec<T> as *mut Vec<T>;
            unsafe {
                let v = (*pptr).remove(0);
                Some(v)
            }
        } else {
            None
        }
    }

    pub fn peek(&self) -> Option<&mut T> {
        if !self.qdata.is_empty() {
            let pptr = &self.qdata[0] as *const T as *mut T;
            unsafe { Some(&mut *pptr) }
        } else {
            None
        }
    }

   pub fn print_type(&self) {
        let _l: &Queue<T> = self;
        print_type_of(self);
   }
}

/*
API: call Seq
Queue::new()
Queue::push()
Queue::peek()
Queue::pop()
Use()
*/

fn main() {
    let mut q: Queue<String> = Queue::new();    // necessary
    q.push(String::from("hello"));              // necessary
    let e = q.peek().unwrap();                  // necessary
    q.print_type();
    println!("{:?}", q);
    println!("{}", *e);
    q.pop();                                    // necessary

    println!("{:?}", q);
    println!("{}", *e); // <- use after free    // necessary
}
```

## Results

### Call graph

Tho following results show the call graph of the target program.
The `main()` function try to call 5 functions (*i.e.,* `new()`, `peek()`, `push()`, `pop()`, `print_type()`), and the `print_type()` only call 1 function (*i.e.,* `print_type_of()`).

```
---------------------------------------
[ROSA] Printing callgraph:
caller: DefId(0:12 ~ queue[768d]::main)
        callee: (bb0, DefId(0:7 ~ queue[768d]::{impl#0}::new))
        callee: (bb3, DefId(0:10 ~ queue[768d]::{impl#0}::peek))
        callee: (bb5, DefId(0:11 ~ queue[768d]::{impl#0}::print_type))
        callee: (bb2, DefId(0:8 ~ queue[768d]::{impl#0}::push))
        callee: (bb12, DefId(0:9 ~ queue[768d]::{impl#0}::pop))
caller: DefId(0:11 ~ queue[768d]::{impl#0}::print_type)
        callee: (bb0, DefId(0:3 ~ queue[768d]::print_type_of))
```

### Unsafe Constructs

Tho following results show the unsafeconstructs of the target program.

`src/main.rs:24` in `pop()` function, `src/main.rs:23` in `pop()` function and `src/main.rs:35` in `peek()` function explicitly use Unsafe construct.

```
---------------------------------------
[ROSA] unsafeconstructs: {
    DefId(0:9 ~ queue[768d]::{impl#0}::pop): {
        src/main.rs:24:17: 26:14 (#0): ExplicitUnsafe(
            HirId {
                owner: DefId(0:9 ~ queue[768d]::{impl#0}::pop),
                local_id: 44,
            },
        ),
        src/main.rs:23:13: 26:14 (#0): ExplicitUnsafe(
            HirId {
                owner: DefId(0:9 ~ queue[768d]::{impl#0}::pop),
                local_id: 44,
            },
        ),
    },
    DefId(0:10 ~ queue[768d]::{impl#0}::peek): {
        src/main.rs:35:13: 35:40 (#0): ExplicitUnsafe(
            HirId {
                owner: DefId(0:10 ~ queue[768d]::{impl#0}::peek),
                local_id: 27,
            },
        ),
    },
}
```

### Raw Pointer

Tho following results show the "raw pointer" information of the target program.

`src/main.rs:34` in `peek()` function and `src/main.rs:22` in `pop()` function try to get the raw pointer.

```
---------------------------------------
fn with rawptrs: 2, rawptrs num: 2, local fn num: 8
[ROSA] rawptrs: {
    DefId(0:10 ~ queue[768d]::{impl#0}::peek): {
        StatementId {
            fn_id: DefId(0:10 ~ queue[768d]::{impl#0}::peek),
            local: _7,
        }: StatementInfo {
            type_name: (
                GetRawPtr,
                "&T",
            ),
            src: Some(
                Init(
                    InitContext {
                        struct_type: "T",
                        fields: "",
                    },
                ),
            ),
            span: src/main.rs:34:24: 34:38 (#0),
            mutability: Not,
            gen_bbs: [
                bb2,
            ],
            kill_bbs: [
                bb4,
            ],
        },
    },
    DefId(0:9 ~ queue[768d]::{impl#0}::pop): {
        StatementId {
            fn_id: DefId(0:9 ~ queue[768d]::{impl#0}::pop),
            local: _8,
        }: StatementInfo {
            type_name: (
                GetRawPtr,
                "&std::vec::Vec<T>",
            ),
            src: Some(
                Init(
                    InitContext {
                        struct_type: "std::vec::Vec<T>",
                        fields: "",
                    },
                ),
            ),
            span: src/main.rs:22:24: 22:35 (#0),
            mutability: Not,
            gen_bbs: [
                bb2,
            ],
            kill_bbs: [
                bb2,
            ],
        },
    },
}
```

### Drop Operation

Tho following results show the `drop()` operation information of the target program.

`src/main.rs:63` in `main()` function and `src/main.rs:67` in `main()` function call the `dorp()` function.

```
---------------------------------------
fn with drop_operations: 1, drop_operations num: 4, local fn num: 8
[ROSA] drop_operations: {
    DefId(0:12 ~ queue[768d]::main): {
        StatementId {
            fn_id: DefId(0:12 ~ queue[768d]::main),
            local: _41,
        }: StatementInfo {
            type_name: (
                Drop,
                "&Queue<std::string::String>",
            ),
            src: Some(
                Init(
                    InitContext {
                        struct_type: "Queue<std::string::String>",
                        fields: "",
                    },
                ),
            ),
            span: src/main.rs:63:12: 63:13 (#0),
            gen_bbs: [
                bb12,
            ],
            kill_bbs: [
                bb12,
                bb13,
            ],
        },
        StatementId {
            fn_id: DefId(0:12 ~ queue[768d]::main),
            local: _5,
        }: StatementInfo {
            type_name: (
                Drop,
                "&mut std::string::String",
            ),
            src: Some(
                LocalSrc(
                    LocalSrcContext {
                        place: "_1",
                    },
                ),
            ),
            span: src/main.rs:67:1: 67:2 (#0),
            gen_bbs: [
                bb3,
            ],
            kill_bbs: [
                bb20,
            ],
        },
    },
}
```
