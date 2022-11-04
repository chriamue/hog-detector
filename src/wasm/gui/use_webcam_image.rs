use image::{DynamicImage, ImageOutputFormat, RgbaImage};
use std::io::Cursor;
use wasm_bindgen::JsCast;
use yew::{html, Callback, Component, Context, Html, Properties};

pub enum Msg {
    GetWebcamImage,
}

pub struct UseWebcamImage {}

#[derive(Clone, PartialEq, Properties)]
pub struct Props {
    pub onchange: Callback<(String, Vec<u8>)>,
}

impl Component for UseWebcamImage {
    type Message = Msg;
    type Properties = Props;

    fn create(_ctx: &Context<Self>) -> Self {
        Self {}
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::GetWebcamImage => {
                let document = web_sys::window().unwrap().document().unwrap();
                let canvas = document.get_element_by_id("canvas").unwrap();
                let canvas: web_sys::HtmlCanvasElement = canvas
                    .dyn_into::<web_sys::HtmlCanvasElement>()
                    .map_err(|_| ())
                    .unwrap();
                let context = canvas
                    .get_context("2d")
                    .unwrap()
                    .unwrap()
                    .dyn_into::<web_sys::CanvasRenderingContext2d>()
                    .unwrap();
                let img = context
                    .get_image_data(
                        0.0 as f64,
                        0.0 as f64,
                        canvas.width() as f64,
                        canvas.height() as f64,
                    )
                    .unwrap();
                let tmp = RgbaImage::from_raw(canvas.width(), canvas.height(), img.data().to_vec())
                    .unwrap();
                let img = DynamicImage::ImageRgba8(tmp);
                let mut image_data: Vec<u8> = Vec::new();
                img.write_to(&mut Cursor::new(&mut image_data), ImageOutputFormat::Png)
                    .unwrap();
                ctx.props()
                    .onchange
                    .emit(("webcam".to_string(), image_data));
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let onclick = ctx.link().callback(|_| Msg::GetWebcamImage);

        html! {
            <>
                <button type="button" class="btn btn-success" {onclick}>
                    { "Get Webcam Image" }
                </button>
            </>
        }
    }
}
