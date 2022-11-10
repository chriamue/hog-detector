#![allow(missing_docs)]
use wasm_bindgen::prelude::*;

pub mod annotation_tool;
pub mod annotations_js;
pub mod download;
pub mod hogdetector_js;
pub mod trainer;

use annotations_js::AnnotationsJS;
use hogdetector_js::HogDetectorJS;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen]
pub fn init_annotation_tool(root: web_sys::Element, annotations: &AnnotationsJS) {
    yew::start_app_with_props_in_element::<annotation_tool::App>(
        root,
        annotation_tool::Props {
            annotations: annotations.clone(),
        },
    );
}

#[wasm_bindgen]
pub fn init_trainer(root: web_sys::Element, annotations: &AnnotationsJS, detector: &HogDetectorJS) {
    yew::start_app_with_props_in_element::<trainer::TrainerApp>(
        root,
        trainer::Props {
            detector: detector.clone(),
            annotations: annotations.clone(),
        },
    );
}
