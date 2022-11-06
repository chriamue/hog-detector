use std::ops::Deref;
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct AnnotationsJS {
    annotations: Arc<Mutex<Vec<String>>>,
}

#[wasm_bindgen]
impl AnnotationsJS {
    #[wasm_bindgen(constructor)]
    pub fn new() -> AnnotationsJS {
        use console_error_panic_hook;
        console_error_panic_hook::set_once();
        AnnotationsJS {
            annotations: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn push(&self, annotation: String) {
        self.annotations.lock().unwrap().push(annotation);
    }
}

impl std::fmt::Display for AnnotationsJS {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let annotations = {
            let annotations = self.annotations.lock().unwrap();
            format!("{:?}", annotations)
        };
        f.pad(&format!("AnnotationsJS {}", annotations))
    }
}

impl PartialEq for AnnotationsJS {
    fn eq(&self, other: &Self) -> bool {
        if ::core::ptr::eq(&self, &other) {
            true
        } else {
            other.annotations.try_lock().unwrap().deref()
                == self.annotations.try_lock().unwrap().deref()
        }
    }
}
