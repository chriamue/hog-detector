use crate::dataset::{DataSet, MemoryDataSet};
use crate::Annotation;
use image::DynamicImage;
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
            self.image.lock().unwrap().clone(),
            self.annotations.lock().unwrap().clone(),
        ));
        dataset.load();
        dataset.generate_random_annotations(10);
        dataset
    }

    pub fn push(&self, annotation: Annotation) {
        self.annotations.lock().unwrap().push(annotation);
    }

    pub fn clear(&self) {
        self.annotations.lock().unwrap().clear();
    }

    pub fn len(&self) -> usize {
        self.annotations.lock().unwrap().len()
    }

    pub fn get_annotations(&self) -> Vec<Annotation> {
        self.annotations.lock().unwrap().clone()
    }

    pub fn get_image(&self) -> DynamicImage {
        self.image.lock().unwrap().clone()
    }
}

#[wasm_bindgen]
impl AnnotationsJS {
    #[wasm_bindgen(constructor)]
    pub fn new() -> AnnotationsJS {
        use console_error_panic_hook;
        console_error_panic_hook::set_once();
        AnnotationsJS {
            image: Arc::new(Mutex::new(DynamicImage::new_rgb8(1, 1))),
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
        ::core::ptr::eq(&self, &other)
    }
}

impl Default for AnnotationsJS {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_len() {
        let annotations = AnnotationsJS::new();
        let annotation = Annotation::default();
        annotations.push(annotation);
        assert_eq!(1, annotations.len());
    }
}
