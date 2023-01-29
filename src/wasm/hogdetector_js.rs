use crate::detector::visualize_detections;
use crate::hogdetector::HogDetectorTrait;
use crate::HogDetector;
use instant::Instant;
use ndarray::{Array1, Array2};
use object_detector_rust::dataset::DataSet;
use object_detector_rust::detector::PersistentDetector;
use object_detector_rust::prelude::Detector;
use object_detector_rust::prelude::MemoryDataSet;
use object_detector_rust::prelude::RandomForestClassifier;
use std::collections::VecDeque;
use std::io::Cursor;
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Clone)]
pub struct HogDetectorJS {
    hog: Arc<Mutex<Box<dyn HogDetectorTrait<f32, usize>>>>,
    timestamps: Arc<Mutex<VecDeque<u128>>>,
}

impl HogDetectorJS {
    pub fn train(&self, dataset: MemoryDataSet) {
        let mut hog = self.hog.lock().unwrap();
        let (x, y) = dataset.get_data();
        let y = y.into_iter().map(|y| y as usize).collect::<Vec<_>>();
        hog.fit_class(&x, &y, 1).unwrap();
    }

    pub fn train_with_hard_negative_samples(&self, dataset: MemoryDataSet) {
        let mut dataset = dataset;
        let mut hog = self.hog.lock().unwrap();
        //dataset.generate_hard_negative_samples(hog.detector(), 1, Some(50));
        dataset.load().unwrap();
        let (x, y) = dataset.get_data();
        let y = y.into_iter().map(|y| y as usize).collect::<Vec<_>>();
        hog.fit_class(&x, &y, 1).unwrap();
    }
}

#[wasm_bindgen]
impl HogDetectorJS {
    #[wasm_bindgen(constructor)]
    pub fn new() -> HogDetectorJS {
        use console_error_panic_hook;
        console_error_panic_hook::set_once();
        let hog = {
            let mut model: HogDetector<f32, usize, RandomForestClassifier<_, _>, _> =
                HogDetector::default();
            let file = Cursor::new(include_bytes!("../../res/eyes_random_forest_model.json"));
            model.load(file).unwrap();
            model
        };
        HogDetectorJS {
            hog: Arc::new(Mutex::new(Box::new(hog))),
            timestamps: Arc::new(Mutex::new(VecDeque::with_capacity(5))),
        }
    }

    #[wasm_bindgen]
    pub fn init_random_forest_classifier(&self) {
        let hog = {
            let mut model: HogDetector<f32, usize, RandomForestClassifier<_, _>, _> =
                HogDetector::default();
            let mut file = Cursor::new(include_bytes!("../../res/eyes_random_forest_model.json"));
            model.load(file).unwrap();
            model
        };
        *self.hog.lock().unwrap() = Box::new(hog);
    }

    #[wasm_bindgen]
    pub fn init_bayes_classifier(&self) {
        let hog = {
            let mut model: HogDetector<f32, usize, RandomForestClassifier<_, _>, _> =
                HogDetector::default();
            let file = Cursor::new(include_bytes!("../../res/eyes_random_forest_model.json"));
            model.load(file).unwrap();
            model
        };
        *self.hog.lock().unwrap() = Box::new(hog);
    }

    #[wasm_bindgen]
    pub fn init_combined_classifier(&self) {
        let hog = {
            let mut model: HogDetector<f32, usize, RandomForestClassifier<_, _>, _> =
                HogDetector::default();
            let file = Cursor::new(include_bytes!("../../res/eyes_random_forest_model.json"));
            model.load(file).unwrap();
            model
        };
        *self.hog.lock().unwrap() = Box::new(hog);
    }

    #[wasm_bindgen]
    pub fn next(&mut self, img_data: &[u8]) -> Vec<u8> {
        let start = Instant::now();
        let mut img =
            image::load_from_memory_with_format(img_data, image::ImageFormat::Png).unwrap();
        img = visualize_detections(&img, &self.hog.lock().unwrap().detect(&img));

        let mut image_data: Vec<u8> = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut image_data),
            image::ImageFormat::Png,
        )
        .unwrap();

        self.timestamps
            .lock()
            .unwrap()
            .push_back(start.elapsed().as_millis());
        image_data
    }

    #[wasm_bindgen]
    pub fn fps(&self) -> f32 {
        let average_millis: f32 = {
            let timestamps = self.timestamps.lock().unwrap();
            println!("{:?}", timestamps);
            if timestamps.len() > 1 {
                let sum: u128 = timestamps.iter().sum();
                sum as f32 / timestamps.len() as f32
            } else {
                1.0
            }
        };
        1000.0 / average_millis
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
