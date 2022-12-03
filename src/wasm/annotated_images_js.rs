use super::annotations_js::AnnotationsJS;
use crate::{
    dataset::{AnnotatedImageSet, DataSet, MemoryDataSet},
    Annotation,
};
use std::{
    ops::Deref,
    sync::{Arc, Mutex},
};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct AnnotatedImagesJS {
    images: Arc<Mutex<Vec<AnnotationsJS>>>,
}

impl AnnotatedImagesJS {
    pub fn push(&self, image: AnnotationsJS) {
        self.images.lock().unwrap().push(image);
    }

    pub fn inner(&self) -> Arc<Mutex<Vec<AnnotationsJS>>> {
        self.images.clone()
    }

    pub fn create_dataset(&self) -> MemoryDataSet {
        let mut dataset = MemoryDataSet::default();
        for annotations in self.images.lock().unwrap().iter() {
            dataset.add_annotated_image((
                annotations.get_image().clone(),
                annotations.get_annotations(),
            ));
        }
        dataset.load();
        dataset.generate_random_annotations(10);
        dataset
    }

    pub fn add_annotation(&self, index: usize, annotation: Annotation) {
        let mut locked = self.images.lock().unwrap();
        let annotations = locked.get_mut(index).unwrap();
        annotations.push(annotation);
    }
}

impl Default for AnnotatedImagesJS {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl AnnotatedImagesJS {
    #[wasm_bindgen(constructor)]
    pub fn new() -> AnnotatedImagesJS {
        AnnotatedImagesJS {
            images: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl PartialEq for AnnotatedImagesJS {
    fn eq(&self, other: &Self) -> bool {
        if ::core::ptr::eq(&self, &other) {
            true
        } else {
            other.images.try_lock().unwrap().deref() == self.images.try_lock().unwrap().deref()
        }
    }
}

impl std::fmt::Display for AnnotatedImagesJS {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let images = {
            let images = self.images.lock().unwrap();
            format!("{:?}", images)
        };
        f.pad(&format!("AnnotatedImagesJS {}", images))
    }
}
