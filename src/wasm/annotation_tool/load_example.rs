use crate::bbox::BBox;
use crate::wasm::AnnotatedImagesJS;
use crate::wasm::AnnotationsJS;

use yew::Callback;
use yew::{html, Component, Context, Html, Properties};

pub enum Msg {
    LoadExample,
}

pub struct LoadExample {}

#[derive(Clone, PartialEq, Properties)]
pub struct Props {
    pub images: AnnotatedImagesJS,
    pub onchange: Callback<Vec<String>>,
}

impl Component for LoadExample {
    type Message = Msg;
    type Properties = Props;

    fn create(_ctx: &Context<Self>) -> Self {
        Self {}
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::LoadExample => {
                let mut annotations = ctx.props().images.clone();
                let labels = load_train_example(&mut annotations);
                ctx.props().onchange.emit(labels);
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let onclick = ctx.link().callback(|_| Msg::LoadExample);

        html! {
            <div id="load-example">
                <button type="button" class="btn btn-success" {onclick}>
                    { "Load Example" }
                </button>
            </div>
        }
    }
}

pub fn load_train_example(annotated_images: &mut AnnotatedImagesJS) -> Vec<String> {
    annotated_images.clear();
    let labels: Vec<String> = include_str!("../../../res/labels.txt")
        .split('\n')
        .map(|s| s.to_string())
        .collect();
    let webcam01_jpg = include_bytes!("../../../res/training/webcam01.jpg");
    let webcam05_jpg = include_bytes!("../../../res/training/webcam05.jpg");
    let webcam06_jpg = include_bytes!("../../../res/training/webcam06.jpg");
    let webcam10_jpg = include_bytes!("../../../res/training/webcam10.jpg");
    let img01 = image::load_from_memory(webcam01_jpg).unwrap();
    let img05 = image::load_from_memory(webcam05_jpg).unwrap();
    let img06 = image::load_from_memory(webcam06_jpg).unwrap();
    let img10 = image::load_from_memory(webcam10_jpg).unwrap();
    let mut annotation01 = AnnotationsJS::new();
    annotation01.set_image(img01);
    let mut annotation05 = AnnotationsJS::new();
    annotation05.set_image(img05);
    let mut annotation06 = AnnotationsJS::new();
    annotation06.set_image(img06);
    let mut annotation10 = AnnotationsJS::new();
    annotation10.set_image(img10);
    let annotations01 = include_bytes!("../../../res/training/webcam01.txt");
    load_annotations(&mut annotation01, annotations01, &labels);
    let annotations05 = include_bytes!("../../../res/training/webcam05.txt");
    load_annotations(&mut annotation05, annotations05, &labels);
    let annotations06 = include_bytes!("../../../res/training/webcam06.txt");
    load_annotations(&mut annotation06, annotations06, &labels);
    let annotations10 = include_bytes!("../../../res/training/webcam10.txt");
    load_annotations(&mut annotation10, annotations10, &labels);

    annotated_images.push(annotation01);
    annotated_images.push(annotation05);
    annotated_images.push(annotation06);
    annotated_images.push(annotation10);
    labels
}

fn load_annotations(annotation_js: &mut AnnotationsJS, data: &[u8], labels: &Vec<String>) {
    let data = std::str::from_utf8(&data).unwrap().to_string();
    for line in data.split('\n').collect::<Vec<&str>>() {
        let mut l = line.split(' ');
        let label = l.next().unwrap();
        let class = labels.iter().position(|x| x == label).unwrap() as u32;
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
        annotation_js.push(annotation);
    }
}
