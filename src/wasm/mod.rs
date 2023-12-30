#![allow(missing_docs)]
use image_label_tool::prelude::*;
use wasm_bindgen::prelude::*;

pub mod display_image;
pub mod hogdetector_js;
pub mod image_queue;
pub mod pipeline;
pub mod trainer;
pub mod video_producer;

use hogdetector_js::HogDetectorJS;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen]
/// init label tool and start app on given root html element
pub async fn init_image_label_tool(root: web_sys::Element, canvas_element_id: String) -> LabelTool {
    let label_tool = LabelTool::new();
    let data = reqwest::get("https://picsum.photos/640/480/")
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    let img = image::load_from_memory(&data).unwrap();
    let annotated_image = AnnotatedImage::new();
    annotated_image.set_image(img);
    label_tool.push(annotated_image);
    image_label_tool::init_label_tool(root, Some(label_tool), Some(canvas_element_id))
}

#[wasm_bindgen]
pub fn init_trainer(root: web_sys::Element, label_tool: &LabelTool, detector: &HogDetectorJS) {
    yew::Renderer::<trainer::TrainerApp>::with_root_and_props(
        root,
        trainer::Props {
            detector: detector.clone(),
            label_tool: label_tool.clone(),
        },
    )
    .render();
}
