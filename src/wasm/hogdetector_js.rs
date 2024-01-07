use crate::classifier::svm::SVMClassifier;
use crate::classifier::BayesClassifier;
use crate::detection_filter::{DetectionFilter, TrackerFilter};
use crate::detector::visualize_detections;
use crate::hogdetector::HogDetectorTrait;
use crate::utils::scale_to_32;
use crate::HogDetector;
use image::DynamicImage;
use instant::Instant;
use object_detector_rust::classifier::CombinedClassifier;
use object_detector_rust::dataset::DataSet;
use object_detector_rust::detector::PersistentDetector;
use object_detector_rust::prelude::MemoryDataSet;
use object_detector_rust::prelude::RandomForestClassifier;
use object_detector_rust::utils::add_hard_negative_samples;
use std::collections::VecDeque;
use std::io::Cursor;
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Clone)]
pub struct HogDetectorJS {
    hog: Arc<Mutex<Box<dyn HogDetectorTrait<f32, usize>>>>,
    timestamps: Arc<Mutex<VecDeque<u128>>>,
    detection_filter: Arc<Mutex<TrackerFilter>>,
}

impl HogDetectorJS {
    pub fn train(&self, dataset: MemoryDataSet) {
        let mut hog = self.hog.lock().unwrap();
        let (x, y) = dataset.get_data();
        let x: Vec<DynamicImage> = x.into_iter().map(scale_to_32).collect();
        let y = y.into_iter().map(|y| y as usize).collect::<Vec<_>>();
        hog.fit_class(&x, &y, 1).unwrap();
    }

    pub fn train_with_hard_negative_samples(&self, dataset: MemoryDataSet) {
        let mut dataset = dataset;
        let mut hog = self.hog.lock().unwrap();

        add_hard_negative_samples(&mut dataset, hog.detector(), 1, Some(50), 32, 32);
        dataset.load().unwrap();
        let (x, y) = dataset.get_data();
        let x = x.into_iter().map(scale_to_32).collect();
        let y = y.into_iter().map(|y| y as usize).collect::<Vec<_>>();
        hog.fit_class(&x, &y, 1).unwrap();
    }

    pub fn get_model(&self) -> &Arc<Mutex<Box<dyn HogDetectorTrait<f32, usize>>>> {
        &self.hog
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
            detection_filter: Arc::new(Mutex::new(TrackerFilter::new(0.2))),
        }
    }

    #[wasm_bindgen]
    pub fn init_random_forest_classifier(&self) {
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
    pub fn init_bayes_classifier(&self) {
        let hog = {
            let mut model: HogDetector<f32, usize, BayesClassifier<_, _>, _> =
                HogDetector::default();
            let file = Cursor::new(include_bytes!("../../res/eyes_bayes_model.json"));
            model.load(file).unwrap();
            model
        };
        *self.hog.lock().unwrap() = Box::new(hog);
    }

    pub fn init_svm_classifier(&self) {
        let hog = {
            let mut model: HogDetector<f32, usize, SVMClassifier, _> = HogDetector::default();
            let file = Cursor::new(include_bytes!("../../res/eyes_svm_burns_model.json"));
            model.load(file).unwrap();
            model
        };
        *self.hog.lock().unwrap() = Box::new(hog);
    }

    #[wasm_bindgen]
    pub fn init_combined_classifier(&self) {
        let hog = {
            let mut model: HogDetector<
                f32,
                usize,
                CombinedClassifier<f32, usize, BayesClassifier<_, _>, RandomForestClassifier<_, _>>,
                _,
            > = HogDetector::default();
            let file = Cursor::new(include_bytes!("../../res/eyes_combined_model.json"));
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
        let detections = &self.hog.lock().unwrap().detect(&img);
        let filtered_detections = &self
            .detection_filter
            .lock()
            .unwrap()
            .filter_detections(&detections);
        img = visualize_detections(&img, filtered_detections);

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
