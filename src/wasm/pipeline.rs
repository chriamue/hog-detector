use crate::{
    detection_filter::TrackerFilter, detector::visualize_detections, hogdetector::HogDetectorTrait,
    prelude::DetectionFilter, HogDetector,
};

use super::image_queue::ImageQueue;
use object_detector_rust::{classifier::RandomForestClassifier, detector::PersistentDetector};
use std::{
    io::Cursor,
    sync::{Arc, Mutex, TryLockError},
};
use web_sys::ImageData;

#[derive(Clone)]
pub struct Pipeline {
    id: usize,
    video_queue: Arc<ImageQueue>,
    processed_queue: Arc<ImageQueue>,
    hog: Arc<Mutex<Box<dyn HogDetectorTrait<f32, usize>>>>,
    detection_filter: Arc<Mutex<TrackerFilter>>,
}

impl PartialEq for Pipeline {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Pipeline {
    pub fn new(video_queue: Arc<ImageQueue>, processed_queue: Arc<ImageQueue>) -> Self {
        let hog = {
            let mut model: HogDetector<f32, usize, RandomForestClassifier<_, _>, _> =
                HogDetector::default();
            let file = Cursor::new(include_bytes!("../../res/eyes_random_forest_model.json"));
            model.load(file).unwrap();
            model
        };
        Pipeline {
            id: rand::random(),
            video_queue,
            processed_queue,
            hog: Arc::new(Mutex::new(Box::new(hog))),
            detection_filter: Arc::new(Mutex::new(TrackerFilter::new(0.2))),
        }
    }

    fn to_dynamic_image(image_data: ImageData) -> image::DynamicImage {
        let img = image::ImageBuffer::from_raw(
            image_data.width(),
            image_data.height(),
            image_data.data().to_vec(),
        )
        .unwrap();
        image::DynamicImage::ImageRgba8(img)
    }

    fn from_dynamic_image(image: image::DynamicImage) -> ImageData {
        let img = image.to_rgba8();
        let width = img.width();
        let height = img.height();
        ImageData::new_with_u8_clamped_array_and_sh(
            wasm_bindgen::Clamped(&img.into_raw().to_vec()),
            width,
            height,
        )
        .unwrap()
    }

    pub fn process(&self) -> Result<(), Box<dyn std::error::Error + '_>> {
        match self.hog.try_lock() {
            Ok(mut processor_guard) => {
                if let Some(mut image_data) = self.video_queue.pop() {

                    let processor = processor_guard.as_mut();
                    let mut image = Pipeline::to_dynamic_image(image_data);
                    let detections = processor.detect(&mut image);
                    let filtered_detections = &self
                        .detection_filter
                        .lock()
                        .unwrap()
                        .filter_detections(&detections);
                    image = visualize_detections(&image, filtered_detections);

                    image_data = Pipeline::from_dynamic_image(image);
                    self.processed_queue.push(image_data)?;
                }
            }
            Err(TryLockError::WouldBlock) => {
                log::warn!("Unable to acquire locks for processor.");
                return Ok(());
            }
            _ => {
                return Err("Failed to acquire necessary locks".into());
            }
        }

        Ok(())
    }
}
