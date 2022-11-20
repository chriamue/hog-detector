use crate::classifier::BayesClassifier;
use crate::classifier::RandomForestClassifier;
use crate::hogdetector::HogDetectorTrait;
use crate::DataSet;
use crate::HogDetector;
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Clone)]
pub struct HogDetectorJS {
    hog: Arc<Mutex<Box<dyn HogDetectorTrait>>>,
}

impl HogDetectorJS {
    pub fn train(&self, dataset: &dyn DataSet) {
        let mut hog = self.hog.lock().unwrap();
        hog.train_class(dataset, 1);
    }
}

#[wasm_bindgen]
impl HogDetectorJS {
    #[wasm_bindgen(constructor)]
    pub fn new() -> HogDetectorJS {
        use console_error_panic_hook;
        console_error_panic_hook::set_once();

        let hog = {
            let model = include_str!("../../res/eyes_random_forest_model.json");
            serde_json::from_str::<HogDetector<RandomForestClassifier>>(model).unwrap()
        };

        HogDetectorJS {
            hog: Arc::new(Mutex::new(Box::new(hog))),
        }
    }

    #[wasm_bindgen]
    pub fn init_random_forest_classifier(&self) {
        let hog = {
            let model = include_str!("../../res/eyes_random_forest_model.json");
            serde_json::from_str::<HogDetector<RandomForestClassifier>>(model).unwrap()
        };
        *self.hog.lock().unwrap() = Box::new(hog);
    }

    #[wasm_bindgen]
    pub fn init_bayes_classifier(&self) {
        let hog = {
            let model = include_str!("../../res/eyes_bayes_model.json");
            serde_json::from_str::<HogDetector<BayesClassifier>>(model).unwrap()
        };
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
