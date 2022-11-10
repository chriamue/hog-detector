use crate::detector::visualize_detections;
use crate::utils::image_to_base64_image;
use crate::wasm::download::download_bytes;
use crate::{prelude::*, Annotation, Class};
use image::DynamicImage;
use yew::{
    events::{DragEvent, MouseEvent},
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
    pos: (i32, i32),
}

#[derive(Clone, PartialEq, Properties)]
pub struct Props {
    pub filename: String,
    pub label: String,
    pub image: DynamicImage,
    pub annotations: Vec<Annotation>,
    pub onchange: Callback<Annotation>,
}

pub fn format_annotation(annotation: &Annotation, labels: &Vec<String>) -> String {
    let (bbox, class) = annotation;
    format!(
        "{} {} {} {} {}",
        labels[*class as usize], bbox.x as i32, bbox.y as i32, bbox.w as i32, bbox.h as i32
    )
}

pub fn format_annotations(annotations: &Vec<Annotation>, labels: &Vec<String>) -> Vec<String> {
    annotations
        .iter()
        .map(|annotation| format_annotation(annotation, labels))
        .collect::<Vec<String>>()
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
            pos: (0, 0),
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::Dropped(_data) => true,
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
                    .unwrap() as Class;
                ctx.props().onchange.emit((
                    BBox {
                        x: x1 as f32,
                        y: y1 as f32,
                        w: (x2 - x1) as f32,
                        h: (y2 - y1) as f32,
                    },
                    class,
                ));
                true
            }
            Msg::DownloadAnnotations => {
                let file_data =
                    format_annotations(&ctx.props().annotations, &self.labels).join("\n");
                download_bytes(
                    file_data.as_bytes(),
                    &format!("{}.txt", ctx.props().filename),
                );
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
        let detections = ctx
            .props()
            .annotations
            .iter()
            .map(|annotation| Detection {
                bbox: annotation.0,
                confidence: 1.0,
                class: annotation.1 as usize,
            })
            .collect();
        let img = visualize_detections(&ctx.props().image, &detections);
        let url = image_to_base64_image(&img);

        html! {
            <div class="flex w-screen bg-gray-100" { ondrop }>
            <img src={url} {onmousedown} {onmousemove} {onmouseup} />
            <p>
            { format_annotations(&ctx.props().annotations, &self.labels).iter().map(|annotation| {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_annotations() {
        let det1 = (
            BBox {
                x: 0.5,
                y: 0.5,
                w: 1.0,
                h: 1.0,
            },
            0,
        );
        let det2 = (
            BBox {
                x: 0.6,
                y: 0.6,
                w: 1.0,
                h: 1.0,
            },
            0,
        );
        let det3 = (
            BBox {
                x: 1.5,
                y: 1.5,
                w: 1.0,
                h: 1.0,
            },
            1,
        );
        let labels = vec!["other".to_string(), "one".to_string()];
        let annotations: Vec<Annotation> = vec![det1, det2, det3];
        let formatted = format_annotations(&annotations, &labels);
        assert_eq!(3, formatted.len());
        assert_eq!("other 0 0 1 1".to_string(), formatted[0]);
    }
}
