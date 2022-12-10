use super::annotated_images_js::AnnotatedImagesJS;
use super::annotations_js::AnnotationsJS;
use crate::prelude::BBox;
use crate::Annotation;
use std::path::Path;
use yew::prelude::*;

pub mod editor;
pub mod header;
pub mod images_list;
pub mod labels;
pub mod load_example;
pub mod upload_annotations;
pub mod upload_image;
pub mod use_webcam_image;

pub enum Msg {
    ExampleLoaded(Vec<String>),
    LabelChanged(String),
    ImageChanged((String, Vec<u8>)),
    AnnotationsChanged((String, Vec<u8>)),
    NewAnnotation(Annotation),
    AddImage(),
    ImageSelected(usize),
}

#[derive(Clone, PartialEq, Properties)]
pub struct Props {
    pub images: AnnotatedImagesJS,
}

pub struct App {
    current: usize,
    current_label: String,
    current_filename: String,
    labels: Vec<String>,
}

impl Component for App {
    type Message = Msg;
    type Properties = Props;

    fn create(_ctx: &Context<Self>) -> Self {
        Self {
            current: 0,
            current_label: "none".to_string(),
            current_filename: "Unknown".to_string(),
            labels: include_str!("../../../res/object_labels.txt")
                .split('\n')
                .map(|s| s.to_string())
                .collect(),
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::ExampleLoaded(labels) => {
                self.labels = labels;
                true
            }
            Msg::LabelChanged(label) => {
                self.current_label = label;
                true
            }
            Msg::ImageChanged((filename, data)) => {
                let img = image::load_from_memory(&data).unwrap();
                self.current_filename = Path::new(&filename)
                    .with_extension("")
                    .to_str()
                    .unwrap()
                    .to_string();
                {
                    let images = ctx.props().images.inner();
                    let mut annotations = images.lock().unwrap();
                    annotations.get_mut(self.current).unwrap().set_image(img);
                    annotations.get_mut(self.current).unwrap().clear()
                };
                true
            }
            Msg::AnnotationsChanged((_, data)) => {
                ctx.props()
                    .images
                    .inner()
                    .lock()
                    .unwrap()
                    .get_mut(self.current)
                    .unwrap()
                    .clear();
                let data = std::str::from_utf8(&data).unwrap().to_string();
                for line in data.split('\n').collect::<Vec<&str>>() {
                    let mut l = line.split(' ');
                    let label = l.next().unwrap();
                    let class = self.labels.iter().position(|x| x == label).unwrap() as u32;
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
                    let annotation = (
                        BBox {
                            x: x as f32,
                            y: y as f32,
                            w: w as f32,
                            h: h as f32,
                        },
                        class,
                    );
                    ctx.props()
                        .images
                        .inner()
                        .lock()
                        .unwrap()
                        .get_mut(self.current)
                        .unwrap()
                        .get_annotations()
                        .push(annotation);
                }
                true
            }
            Msg::NewAnnotation(annotation) => {
                ctx.props().images.add_annotation(self.current, annotation);
                true
            }
            Msg::AddImage() => {
                let img = image::DynamicImage::new_rgb8(1, 1);
                let annotations = AnnotationsJS::new();
                annotations.set_image(img);
                ctx.props().images.push(annotations);
                true
            }
            Msg::ImageSelected(index) => {
                self.current = index;
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let label = self.current_label.clone();
        let on_label_change = ctx.link().callback(Msg::LabelChanged);
        let on_image_change = ctx
            .link()
            .callback(|(filename, data): (String, Vec<u8>)| Msg::ImageChanged((filename, data)));
        let on_annotations_change = ctx.link().callback(|(filename, data): (String, Vec<u8>)| {
            Msg::AnnotationsChanged((filename, data))
        });
        let on_example_loaded = ctx.link().callback(|labels| Msg::ExampleLoaded(labels));
        let on_new_annotation = ctx.link().callback(Msg::NewAnnotation);
        let on_add_image = ctx.link().callback(|()| Msg::AddImage());
        let on_image_selected = ctx.link().callback(Msg::ImageSelected);
        let images = ctx.props().images.inner().lock().unwrap().clone();
        let (image, annotations) = {
            let images = ctx.props().images.inner();
            let image_annotations = images.lock().unwrap();
            let image_annotations = image_annotations.get(self.current).unwrap();
            let image = image_annotations.get_image();
            let annotations = image_annotations.get_annotations();
            (image, annotations)
        };
        html! {
            <>
            <header::Header />
            <div id="data-sources">
            <use_webcam_image::UseWebcamImage onchange={on_image_change.clone()} />
            <upload_image::UploadImage onchange={on_image_change}/>
            <upload_annotations::UploadAnnotations onchange={on_annotations_change}/>
            <load_example::LoadExample images={ctx.props().images.clone()} onchange={on_example_loaded} />
            </div>
            <labels::Labels onchange={ on_label_change } labels={self.labels.clone()} label={ label.clone()} />
            <div id="annotation-tool-container">
            <editor::Editor {label} filename={self.current_filename.to_string()} {image} {annotations} onchange={on_new_annotation}/>
            <images_list::ImagesList {images} onaddimage={on_add_image} current={self.current} onimageselected={on_image_selected} />
            </div>
            </>
        }
    }
}
