use hog_detector::wasm::display_image::DisplayImage;
use hog_detector::wasm::hogdetector_js::HogDetectorJS;
use hog_detector::wasm::image_queue::ImageQueue;
use hog_detector::wasm::init_image_label_tool;
use hog_detector::wasm::pipeline::Pipeline;
use hog_detector::wasm::trainer::TrainerApp;
use hog_detector::wasm::video_producer::VideoProducer;
use image_label_tool::prelude::*;
use log::Level;
use std::sync::Arc;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;
use web_sys::wasm_bindgen::UnwrapThrowExt;
use web_sys::{Element, Window};
use yew::prelude::*;

#[function_component]
pub fn App() -> Html {
    let video_queue = use_state(|| Arc::new(ImageQueue::new_with_id(1, 3)));
    let processed_queue = use_state(|| Arc::new(ImageQueue::new(3)));

    let pipeline = use_state(|| {
        Arc::new(Pipeline::new(
            (*video_queue).clone(),
            (*processed_queue).clone(),
        ))
    });

    let detector = use_state(|| HogDetectorJS::new());
    let label_tool = use_state(|| LabelTool::new());

    let canvas_ref = use_node_ref();

    use_effect(move || {
        let pipeline_clone = pipeline.clone();
        let window: Window = web_sys::window().unwrap_throw();
        let closure = Closure::wrap(Box::new(move || {
            if let Err(e) = pipeline_clone.process() {
                log::error!("Error processing pipeline: {:?}", e);
            }
        }) as Box<dyn FnMut()>);

        let interval_id = window
            .set_interval_with_callback_and_timeout_and_arguments_0(
                closure.as_ref().unchecked_ref(),
                10,
            )
            .unwrap_throw();

        closure.forget();

        move || {
            window.clear_interval_with_handle(interval_id);
        }
    });

    let cloned_label_tool = label_tool.clone();
    let canvas_ref_clone = canvas_ref.clone();

    use_effect_with(canvas_ref_clone.clone(), |_| {
        wasm_bindgen_futures::spawn_local(async move {
            let root = canvas_ref_clone.cast::<Element>().unwrap_throw();
            let tool = init_image_label_tool(root, "video-producer-canvas".to_string()).await;
            cloned_label_tool.set(tool);
        });
    });

    html! {
        <div>
        <VideoProducer image_queue={(*video_queue).clone()} />
        <DisplayImage image_queue={(*processed_queue).clone()} />
        <TrainerApp label_tool={(*label_tool).clone()} detector={(*detector).clone()} />
        <div ref={canvas_ref} id="annotation-tool" />
        </div>
    }
}

fn main() {
    let _ = console_log::init_with_level(Level::Debug);
    yew::Renderer::<App>::new().render();
}
