use crate::classifier::BayesClassifier;
use crate::classifier::CombinedClassifier;
use crate::classifier::RandomForestClassifier;
use crate::dataset::DataGenerator;
use crate::dataset::DataSet;
use crate::dataset::MemoryDataSet;
use crate::hogdetector::HogDetectorTrait;
use crate::HogDetector;
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Clone)]
pub struct HogDetectorJS {
    hog: Arc<Mutex<Box<dyn HogDetectorTrait>>>,
}

impl HogDetectorJS {
    pub fn train(&self, dataset: MemoryDataSet) {
        let mut hog = self.hog.lock().unwrap();
        hog.train_class(&dataset, 1);
    }

    pub fn train_with_hard_negative_samples(&self, dataset: MemoryDataSet) {
        let mut dataset = dataset;
        let mut hog = self.hog.lock().unwrap();
        hog.train_class(&dataset, 1);
        dataset.generate_hard_negative_samples(hog.detector(), 1, Some(50));
        dataset.load();
        hog.train_class(&dataset, 1);
    }
}

#[wasm_bindgen]
impl HogDetectorJS {
    #[wasm_bindgen(constructor)]
    pub fn new() -> HogDetectorJS {
        use console_error_panic_hook;
        console_error_panic_hook::set_once();

        let mut hog = HogDetector::<RandomForestClassifier>::default();
        let model = include_str!("../../res/eyes_random_forest_model.json");
        hog.load(model);

        HogDetectorJS {
            hog: Arc::new(Mutex::new(Box::new(hog))),
        }
    }

    #[wasm_bindgen]
    pub fn init_random_forest_classifier(&self) {
        let mut hog = HogDetector::<RandomForestClassifier>::default();
        let model = include_str!("../../res/eyes_random_forest_model.json");
        hog.load(model);
        *self.hog.lock().unwrap() = Box::new(hog);
    }

    #[wasm_bindgen]
    pub fn init_bayes_classifier(&self) {
        let mut hog = HogDetector::<BayesClassifier>::default();
        let model = include_str!("../../res/eyes_bayes_model.json");
        hog.load(model);
        *self.hog.lock().unwrap() = Box::new(hog);
    }

    #[wasm_bindgen]
    pub fn init_combined_classifier(&self) {
        let mut hog = HogDetector::<CombinedClassifier>::default();
        let model = include_str!("../../res/eyes_combined_model.json");
        hog.load(model);
        *self.hog.lock().unwrap() = Box::new(hog);
    }

    #[wasm_bindgen]
    pub fn next(&mut self, img_data: &[u8]) -> Vec<u8> {
        let mut img =
            image::load_from_memory_with_format(img_data, image::ImageFormat::Png).unwrap();

        img = self.hog.lock().unwrap().visualize_detections(&img);

        let mut image_data: Vec<u8> = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut image_data),
            image::ImageFormat::Png,
        )
        .unwrap();
        image_data
    }
}

impl PartialEq for HogDetectorJS {
    fn eq(&self, other: &Self) -> bool {
        ::core::ptr::eq(&self, &other)
    }
}

impl Default for HogDetectorJS {
    fn default() -> Self {
        Self::new()
    }
}
