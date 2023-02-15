pub mod queue {

    #[derive(Debug)]
    pub struct Queue {
        qdata: Vec<String>,
    }

    impl Queue {
        pub fn new() -> Self {
            Queue { qdata: Vec::new() }
        }

        pub fn push(&mut self, item: String) {
            self.qdata.push(item);
        }

        pub fn pop(&self) -> Option<String> {
            let l = self.qdata.len();
            if l > 0 {
                let raw = &self.qdata as *const Vec<String> as *mut Vec<String>;
                unsafe {
                    let v = (*raw).remove(0);
                    Some(v)
                }
            } else {
                None
            }
        }

        pub fn peek(&self) -> Option<&mut String> {
            if !self.qdata.is_empty() {
                let raw = &self.qdata[0] as *const String as *mut String;
                unsafe { Some(&mut *raw) }
            } else {
                None
            }
        }
    }
}
