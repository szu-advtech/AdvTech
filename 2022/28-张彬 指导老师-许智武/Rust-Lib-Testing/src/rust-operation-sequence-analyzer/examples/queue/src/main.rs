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

