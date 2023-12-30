use std::error::Error;
use std::sync::Arc;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue, UnwrapThrowExt};
use wasm_bindgen_futures::JsFuture;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, ImageData};
use web_sys::{HtmlVideoElement, MediaStreamConstraints};
use yew::prelude::*;

use super::image_queue::ImageQueue;

async fn get_user_media() -> Result<web_sys::MediaStream, JsValue> {
    let window = web_sys::window().unwrap_throw();
    let navigator = window.navigator();
    let media_devices = navigator.media_devices()?;
    let mut constraints = MediaStreamConstraints::new();
    constraints.video(&JsValue::from(true));

    let media_stream_promise = media_devices.get_user_media_with_constraints(&constraints)?;
    JsFuture::from(media_stream_promise)
        .await
        .map(|stream| stream.unchecked_into())
}

async fn init_webcam(video: HtmlVideoElement) -> Result<(), JsValue> {
    let media_stream = get_user_media().await?;
    video.set_src_object(Some(&media_stream));
    video.set_attribute("autoplay", "true")?;
    video.set_attribute("id", "video")?;
    video.set_attribute("style", "display: none").unwrap();
    Ok(())
}

fn draw_video_on_canvas(video: &HtmlVideoElement, canvas: &HtmlCanvasElement) {
    let context = canvas
        .get_context("2d")
        .unwrap_throw()
        .unwrap_throw()
        .dyn_into::<CanvasRenderingContext2d>()
        .unwrap_throw();

    // Set the dimensions of the drawing surface (not just the display size)
    canvas.set_width(video.video_width());
    canvas.set_height(video.video_height());

    context
        .draw_image_with_html_video_element(video, 0.0, 0.0)
        .unwrap_throw();
}

fn capture_frame_as_image(
    video: &HtmlVideoElement,
    canvas: &HtmlCanvasElement,
) -> Result<ImageData, Box<dyn Error>> {
    draw_video_on_canvas(video, canvas);
    let context = canvas
        .get_context("2d")
        .map_err(|e| e.as_string().unwrap_or("Unknown error".to_string()))?
        .ok_or("Failed to get 2D context")?
        .dyn_into::<CanvasRenderingContext2d>()
        .map_err(|_| "Failed to cast to CanvasRenderingContext2d")?;

    context
        .get_image_data(0.0, 0.0, canvas.width() as f64, canvas.height() as f64)
        .map_err(|e| {
            e.as_string()
                .unwrap_or("Failed to get image data".to_string())
                .into()
        })
}

fn producer_task(
    video: &HtmlVideoElement,
    canvas: &HtmlCanvasElement,
    image_queue: Arc<ImageQueue>,
) {
    if let Ok(frame) = capture_frame_as_image(&video, &canvas) {
        match image_queue.push(frame) {
            Ok(_) => (),
            Err(e) => log::warn!("Producer: Failed to add frame - {}", e),
        }
    }
}

#[derive(Properties, PartialEq, Clone)]
pub struct VideoProducerProps {
    pub image_queue: Arc<ImageQueue>,
}
#[function_component(VideoProducer)]
pub fn video_producer(props: &VideoProducerProps) -> Html {
    let video_ref = use_node_ref();
    let canvas_ref = use_node_ref();

    let video_ref_clone = video_ref.clone();
    let canvas_ref_clone = canvas_ref.clone();
    let frame_queue_producer = props.image_queue.clone();

    use_effect_with(frame_queue_producer.clone(), move |_| {
        let video = video_ref_clone.cast::<HtmlVideoElement>().unwrap_throw();
        let canvas = canvas_ref_clone.cast::<HtmlCanvasElement>().unwrap_throw();

        let cloned_video = video.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if let Err(e) = init_webcam(cloned_video.clone()).await {
                log::error!("Error initializing webcam: {:?}", e);
            }
        });

        let callback = Closure::wrap(Box::new(move || {
            let frame_queue_producer = frame_queue_producer.clone();
            producer_task(&video, &canvas, frame_queue_producer);
        }) as Box<dyn Fn()>);

        let handle = web_sys::window()
            .unwrap_throw()
            .set_interval_with_callback_and_timeout_and_arguments_0(
                callback.as_ref().unchecked_ref(),
                10,
            )
            .unwrap_throw();

        callback.forget();

        move || {
            web_sys::window()
                .unwrap_throw()
                .clear_interval_with_handle(handle);
        }
    });

    html! {
        <div>
            <video ref={video_ref} autoplay=true />
            <canvas id="video-producer-canvas" ref={canvas_ref} style="display: none;" />
        </div>
    }
}
