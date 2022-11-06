use super::AnnotationsJS;
use crate::prelude::BBox;
use crate::prelude::Detection;
use image::{DynamicImage, ImageOutputFormat};
use std::io::Cursor;
use std::path::Path;
use yew::prelude::*;

pub mod editor;
pub mod header;
pub mod labels;
pub mod upload_annotations;
pub mod upload_image;
pub mod use_webcam_image;

pub fn image_to_base64(img: &DynamicImage) -> String {
    let mut image_data: Vec<u8> = Vec::new();
    img.write_to(&mut Cursor::new(&mut image_data), ImageOutputFormat::Png)
        .unwrap();
    let res_base64 = base64::encode(image_data);
    format!("data:image/png;base64,{}", res_base64)
}

pub enum Msg {
    LabelChanged(String),
    ImageChanged((String, Vec<u8>)),
    AnnotationsChanged((String, Vec<u8>)),
    NewAnnotation(Detection),
}

#[derive(Clone, PartialEq, Properties)]
pub struct Props {
    pub annotations: AnnotationsJS,
}

pub struct App {
    current_label: String,
    current_image: String,
    current_filename: String,
    labels: Vec<String>,
    annotations: Vec<Detection>,
}

impl Component for App {
    type Message = Msg;
    type Properties = Props;

    fn create(_ctx: &Context<Self>) -> Self {
        Self {
            current_label: "none".to_string(),
            current_image: "https://picsum.photos/640/480/".to_string(),
            current_filename: "Unknown".to_string(),
            labels: include_str!("../../../res/labels.txt")
                .split("\n")
                .map(|s| s.to_string())
                .collect(),
            annotations: Vec::new(),
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::LabelChanged(label) => {
                self.current_label = label;
                true
            }
            Msg::ImageChanged((filename, data)) => {
                let img = image::load_from_memory(&data.clone()).unwrap();
                self.current_image = image_to_base64(&img);
                self.annotations.clear();
                self.current_filename = format!(
                    "{}",
                    Path::new(&filename).with_extension("").to_str().unwrap()
                );
                true
            }
            Msg::AnnotationsChanged((_, data)) => {
                self.annotations.clear();
                let data = std::str::from_utf8(&data).unwrap().to_string();
                for line in data.split('\n').collect::<Vec<&str>>() {
                    let mut l = line.split(' ');
                    let label = l.next().unwrap();
                    let class = self.labels.iter().position(|x| x == label).unwrap() as usize;
                    let x: u32 = l.next().unwrap().parse().unwrap();
                    let y: u32 = l.next().unwrap().parse().unwrap();
                    let w: u32 = match l.next() {
                        Some(w) => w.parse().unwrap(),
                        None => 32,
                    };
                    let h: u32 = match l.next() {
                        Some(h) => h.parse().unwrap(),
                        None => 32,
                    };
                    let annotation = Detection {
                        class,
                        confidence: 1.0,
                        bbox: BBox {
                            x: x as f32,
                            y: y as f32,
                            w: w as f32,
                            h: h as f32,
                        },
                    };
                    self.annotations.push(annotation);
                }
                true
            }
            Msg::NewAnnotation(annotation) => {
                self.annotations.push(annotation);
                let formatted = editor::format_annotation(&annotation, &self.labels);
                ctx.props().annotations.push(formatted);
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
        let on_annotations_change = ctx.link().callback(|(filename, data): (String, Vec<u8>)| {
            Msg::AnnotationsChanged((filename, data))
        });
        let on_new_annotation = ctx
            .link()
            .callback(|annotation: Detection| Msg::NewAnnotation(annotation));
        html! {
            <>
            <header::Header />
            <use_webcam_image::UseWebcamImage onchange={on_image_change.clone()} />
            <upload_image::UploadImage onchange={on_image_change.clone()}/>
            <upload_annotations::UploadAnnotations onchange={on_annotations_change}/>
            <labels::Labels onchange={ on_label_change } label={ label.clone()} />
            <editor::Editor {label} filename={self.current_filename.to_string()} image={self.current_image.to_string()} annotations={self.annotations.clone()} onchange={on_new_annotation}/>
            </>
        }
    }
}
