pub mod queue {

    #[derive(Debug)]
    pub struct Queue<T> {
        qdata: Vec<T>,
    }
	
	fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

    impl<T> Queue<T> {
        pub fn new() -> Self {
            Queue { qdata: Vec::new() }
        }

        pub fn push(&mut self, item: T) {
            self.qdata.push(item);
        }

        pub fn pop(&self) -> Option<T> {
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
		
		pub fn print_type(&self) {
			let _l: &Queue<T> = self;
			print_type_of(self);
		}
    }
}
