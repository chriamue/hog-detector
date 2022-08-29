use crate::prelude::*;
use crate::wasm::download::download_bytes;
use yew::{
    events::{DragEvent, Event, MouseEvent},
    html, Callback, Component, Context, Html, Properties,
};

pub enum Msg {
    Dropped(String),
    MouseDown(i32, i32),
    MouseUp(i32, i32),
    DownloadAnnotations,
    Nothing,
}

pub struct Editor {
    labels: Vec<String>,
    src: String,
    pos: (i32, i32),
    annotations: Vec<Detection>,
}

#[derive(Clone, PartialEq, Properties)]
pub struct Props {
    pub label: String,
}

impl Editor {
    fn format_annotations(&self) -> Vec<String> {
        self.annotations
            .iter()
            .map(|annotation| {
                format!(
                    "{} {} {} {} {}",
                    self.labels[annotation.class],
                    annotation.bbox.x as i32,
                    annotation.bbox.y as i32,
                    annotation.bbox.w as i32,
                    annotation.bbox.h as i32
                )
            })
            .collect::<Vec<String>>()
    }
}

impl Component for Editor {
    type Message = Msg;
    type Properties = Props;

    fn create(_ctx: &Context<Self>) -> Self {
        Self {
            labels: include_str!("../../../res/labels.txt")
                .split("\n")
                .map(|s| s.to_string())
                .collect(),
            src: "https://picsum.photos/640/480/".to_string(),
            pos: (0, 0),
            annotations: Vec::new(),
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::Dropped(data) => true,
            Msg::MouseDown(x1, y1) => {
                self.pos = (x1, y1);
                true
            }
            Msg::MouseUp(x2, y2) => {
                let (x1, y1) = self.pos;
                let class = self
                    .labels
                    .iter()
                    .position(|x| x == &ctx.props().label)
                    .unwrap() as usize;
                self.annotations.push(Detection {
                    class,
                    confidence: 1.0,
                    bbox: BBox {
                        x: x1 as f32,
                        y: y1 as f32,
                        w: (x2 - x1) as f32,
                        h: (y2 - y1) as f32,
                    },
                });
                true
            }
            Msg::DownloadAnnotations => {
                let file_data = self.format_annotations().join("\n");
                download_bytes(file_data.as_bytes(), "annotations.txt");
                true
            }
            _ => false,
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let onclick = ctx.link().callback(|_| Msg::DownloadAnnotations);
        let ondrop = ctx
            .link()
            .callback(|_: DragEvent| Msg::Dropped("dropped".to_string()));
        let onmousedown = ctx
            .link()
            .callback(|e: MouseEvent| Msg::MouseDown(e.offset_x(), e.offset_y()));
        let onmousemove = ctx.link().callback(|e: MouseEvent| {
            e.prevent_default();
            Msg::Nothing
        });
        let onmouseup = ctx
            .link()
            .callback(|e: MouseEvent| Msg::MouseUp(e.offset_x(), e.offset_y()));
        html! {
            <div class="flex w-screen bg-gray-100" { ondrop }>
            <img src={self.src.to_string()} {onmousedown} {onmousemove} {onmouseup} />
            <p>
            { self.format_annotations().iter().map(|annotation| {
                    html!{<>{annotation}<br/></>}
            }).collect::<Html>() }
            </p>
            <button type="button" class="btn btn-success" {onclick}>
                { "Download Annotations" }
            </button>
            </div>
        }
    }
}
