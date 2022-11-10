#![allow(missing_docs)]
use wasm_bindgen::prelude::*;

pub mod annotated_images_js;
pub mod annotation_tool;
pub mod annotations_js;
pub mod download;
pub mod hogdetector_js;
pub mod trainer;

use annotated_images_js::AnnotatedImagesJS;
use annotations_js::AnnotationsJS;
use hogdetector_js::HogDetectorJS;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen]
pub async fn init_images(images: &AnnotatedImagesJS) {
    let data = reqwest::get("https://picsum.photos/640/480/")
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    let img = image::load_from_memory(&data).unwrap();
    let annotations = AnnotationsJS::new();
    annotations.set_image(img);
    images.push(annotations);
}

#[wasm_bindgen]
pub fn init_annotation_tool(root: web_sys::Element, images: &AnnotatedImagesJS) {
    yew::start_app_with_props_in_element::<annotation_tool::App>(
        root,
        annotation_tool::Props {
            images: images.clone(),
        },
    );
}

#[wasm_bindgen]
pub fn init_trainer(root: web_sys::Element, images: &AnnotatedImagesJS, detector: &HogDetectorJS) {
    yew::start_app_with_props_in_element::<trainer::TrainerApp>(
        root,
        trainer::Props {
            detector: detector.clone(),
            images: images.clone(),
        },
    );
}
