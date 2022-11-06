use crate::Detector;
use crate::HogDetector;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

pub mod annotations_js;
pub mod download;
pub mod gui;
pub mod trainer;

use annotations_js::AnnotationsJS;
use wasm_bindgen_test::console_log;

#[wasm_bindgen]
#[derive(PartialEq, Clone)]
pub struct HogDetectorJS {
    hog: Arc<HogDetector>,
}

#[wasm_bindgen]
impl HogDetectorJS {
    #[wasm_bindgen(constructor)]
    pub fn new() -> HogDetectorJS {
        use console_error_panic_hook;
        console_error_panic_hook::set_once();

        let hog = {
            let model = include_str!("../../res/eyes_model.json");
            serde_json::from_str::<HogDetector>(&model).unwrap()
        };

        HogDetectorJS { hog: Arc::new(hog) }
    }

    #[wasm_bindgen]
    pub fn next(&mut self, img_data: &[u8]) -> Vec<u8> {
        let mut img =
            image::load_from_memory_with_format(img_data, image::ImageFormat::Png).unwrap();

        img = self.hog.visualize_detections(&mut img);

        let mut image_data: Vec<u8> = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut image_data),
            image::ImageFormat::Png,
        )
        .unwrap();
        image_data
    }
}

#[wasm_bindgen]
pub fn main(root: web_sys::Element, annotations: &mut AnnotationsJS) {
    yew::start_app_with_props_in_element::<gui::App>(
        root,
        gui::Props {
            annotations: annotations.clone(),
        },
    );
}

#[wasm_bindgen]
pub fn init_trainer(
    root: web_sys::Element,
    annotations: &mut AnnotationsJS,
    detector: &HogDetectorJS,
) {
    console_log!("{}", annotations);
    yew::start_app_with_props_in_element::<trainer::TrainerApp>(
        root,
        trainer::Props {
            detector: detector.clone(),
            annotations: annotations.clone(),
        },
    );
}
