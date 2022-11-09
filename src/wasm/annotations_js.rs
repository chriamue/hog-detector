use crate::dataset::{DataSet, MemoryDataSet};
use crate::Annotation;
use image::DynamicImage;
use std::ops::Deref;
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct AnnotationsJS {
    image: Arc<Mutex<DynamicImage>>,
    annotations: Arc<Mutex<Vec<Annotation>>>,
}

impl AnnotationsJS {
    pub fn set_image(&self, image: DynamicImage) {
        *self.image.lock().unwrap() = image;
    }

    pub fn create_dataset(&self) -> MemoryDataSet {
        let mut dataset = MemoryDataSet::default();
        dataset.add((
            self.image.lock().unwrap().to_rgb8(),
            self.annotations.lock().unwrap().clone(),
        ));
        dataset.load(false);
        dataset.generate_random_annotations(10);
        dataset
    }

    pub fn push(&self, annotation: Annotation) {
        self.annotations.lock().unwrap().push(annotation);
    }
}

#[wasm_bindgen]
impl AnnotationsJS {
    #[wasm_bindgen(constructor)]
    pub fn new() -> AnnotationsJS {
        use console_error_panic_hook;
        console_error_panic_hook::set_once();
        AnnotationsJS {
            image: Arc::new(Mutex::new(DynamicImage::default())),
            annotations: Arc::new(Mutex::new(Vec::new())),
        }
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
