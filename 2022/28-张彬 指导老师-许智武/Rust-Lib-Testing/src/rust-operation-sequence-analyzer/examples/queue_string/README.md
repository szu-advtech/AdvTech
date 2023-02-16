## queue_unsafe

To Reproduce this Bug via AddressSanitizer

```sh
cd queue_unsafe
export RUSTFLAGS=-Zsanitizer=address RUSTDOCFLAGS=-Zsanitizer=address
cargo +nightly run
# ensure that you have install nightly Rust toolchain (try to execute command "rustup install nightly")
```

Function `peek()` returns a reference of the object at the head of a queue, and `pop()` pops (removes) the head object from the queue. A use-after-free error may happen with the following sequence of operations (all safe code): a program first calls `peek()` and saves the returned reference at line 5, then calls `pop()` and drops the returned object at line 6, and finally uses the previously saved reference to access the (dropped) object at line 7. This potential error is caused by holding an immutable reference while changing the underlying object. This operation is allowed by Rust because both functions take an immutable reference `&self` as input. When these functions are called, the ownership of the queue is immutably borrowed to both functions.

```rust{.line-numbers}
impl<T, ...> Queue<T, ...> {
    pub fn pop(&self) -> Option<T>{ unsafe {...}}
    pub fn peek(&self) -> Option<&mut T> {unsafe {...}}
}

// POC
fn main() {
    // ...
    let e = Q.peek().unwrap()
    {Q.pop()}
    println!("{}", *e); // <- use after free
}
```

According to the program semantics, `pop()` actually changes the immutably borrowed queue. This interior mutability is improperly written, which results in the potential error. An easy way to avoid this error is to change the input parameter of `pop()` to `&mut self`. When a queue is immutably borrowed by `peek()` at line 5, the borrowing does not end until line 7, since the default lifetime rule extends the lifetime of &self to the lifetime of the returned reference. After the change, the Rust compiler will not allow the mutable borrow by `pop()` at line 6.

Full version: [Permalink to the playground](https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=69c1d4c4901f7991e9e204c14cd52d1d)

```rust{.line-numbers}
#[derive(Debug)]
struct Queue<T> {
    qdata: Vec<T>,
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
            let raw = &self.qdata as *const Vec<T> as *mut Vec<T>;
            unsafe {
                let v = (*raw).remove(0);
                Some(v)
            }
        } else {
            None
        }
    }

    pub fn peek(&self) -> Option<&mut T> {
        if !self.qdata.is_empty() {
            let raw = &self.qdata[0] as *const T as *mut T;
            unsafe { Some(&mut *raw) }
        } else {
            None
        }
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
    
    println!("{:?}", q);
    println!("{}", *e);
    q.pop();                                    // necessary

    println!("{:?}", q);
    println!("{}", *e); // <- use after free    // necessary
}
```
