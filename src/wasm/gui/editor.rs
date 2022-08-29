use crate::prelude::*;
use yew::{
    events::{DragEvent, MouseEvent},
    html, Callback, Component, Context, Html, Properties,
};

pub enum Msg {
    Dropped(String),
    MouseDown(i32, i32),
    MouseUp(i32, i32),
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
            _ => false,
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
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
            { self.annotations.iter().map(|annotation| {
                    html!{<>{format!("{} {} {} {} {}", self.labels[annotation.class],
                        annotation.bbox.x as i32, annotation.bbox.y as i32, annotation.bbox.w as i32, annotation.bbox.h as i32)}<br/></>}
            }).collect::<Html>() }
            </p>
            </div>
        }
    }
}
