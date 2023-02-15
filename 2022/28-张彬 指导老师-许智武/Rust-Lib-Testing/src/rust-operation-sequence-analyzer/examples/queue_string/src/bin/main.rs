use queue_string::queue::Queue;

/*
API: call Seq
Queue::new()
Queue::push()
Queue::peek()
Queue::pop()
Use()
*/

fn main() {
    let mut q: Queue = Queue::new();    // necessary
    q.push(String::from("hello"));              // necessary
    let e = q.peek().unwrap();                  // necessary
    
    println!("{:?}", q);
    println!("{}", *e);
    q.pop();                                    // necessary

    println!("{:?}", q);
    println!("{}", *e); // <- use after free    // necessary
}
