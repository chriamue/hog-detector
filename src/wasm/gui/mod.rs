use image::{DynamicImage, ImageOutputFormat};
use std::io::Cursor;
use yew::prelude::*;

pub mod editor;
pub mod header;
pub mod labels;
pub mod upload;

fn image_to_base64(img: &DynamicImage) -> String {
    let mut image_data: Vec<u8> = Vec::new();
    img.write_to(&mut Cursor::new(&mut image_data), ImageOutputFormat::Png)
        .unwrap();
    let res_base64 = base64::encode(image_data);
    format!("data:image/png;base64,{}", res_base64)
}

pub enum Msg {
    LabelChanged(String),
    ImageChanged((String, Vec<u8>)),
}

pub struct App {
    current_label: String,
    current_image: String,
}

impl Component for App {
    type Message = Msg;
    type Properties = ();

    fn create(_ctx: &Context<Self>) -> Self {
        Self {
            current_label: "none".to_string(),
            current_image: "https://picsum.photos/640/480/".to_string(),
        }
    }

    fn update(&mut self, _ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::LabelChanged(label) => {
                self.current_label = label;
                true
            }
            Msg::ImageChanged((filename, data)) => {
                let img = image::load_from_memory(&data.clone()).unwrap();
                self.current_image = image_to_base64(&img);
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let label = self.current_label.clone();
        let on_label_change = ctx
            .link()
            .callback(|label: String| Msg::LabelChanged(label.to_string()));
        let on_image_change = ctx
            .link()
            .callback(|(filename, data): (String, Vec<u8>)| Msg::ImageChanged((filename, data)));
        html! {
            <>
            <header::Header />
            <upload::Upload onchange={on_image_change}/>
            <labels::Labels onchange={ on_label_change } label={ label.clone()} />
            <editor::Editor {label} image={self.current_image.to_string()}/>
            </>
        }
    }
}
