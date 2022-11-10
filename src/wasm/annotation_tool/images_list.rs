use crate::{utils::image_to_base64_image, wasm::annotations_js::AnnotationsJS};
use yew::{html, Component, Context, Html, Properties};

pub struct ImagesList;

#[derive(Clone, PartialEq, Properties)]
pub struct Props {
    pub image_annotations: Vec<AnnotationsJS>,
}

impl Component for ImagesList {
    type Message = ();
    type Properties = Props;

    fn create(_ctx: &Context<Self>) -> Self {
        Self
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        html! {
            <div class="card" style="width: 18rem;">
            <ul class="list-group list-group-flush">
            {
                ctx.props().image_annotations.clone().into_iter().enumerate().map(|(i, annotations)| {
                    html!{<div key={format!("annotations-{}",i)}>
                    <img src={image_to_base64_image(&annotations.get_image())} width={32} height={32} />
                    { format!("count: {}",annotations.len()) }
                    </div>}
                }).collect::<Html>()
            }
            </ul>
          </div>
        }
    }
}
