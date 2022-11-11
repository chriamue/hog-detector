use crate::{utils::image_to_base64_image, wasm::annotations_js::AnnotationsJS};
use yew::{html, Callback, Component, Context, Html, Properties};

pub struct ImagesList;

pub enum Msg {
    AddImage,
}

#[derive(Clone, PartialEq, Properties)]
pub struct Props {
    pub images: Vec<AnnotationsJS>,
    pub current: usize,
    pub onaddimage: Callback<()>,
    pub onimageselected: Callback<usize>,
}

impl ImagesList {
    fn create_image_element(
        &self,
        index: usize,
        annotations: &AnnotationsJS,
        current: usize,
        onclick: Callback<usize>,
    ) -> Html {
        let border = if index == current {
            "border border-primary"
        } else {
            ""
        };
        html! {<li class="list-group-item">
        <div key={format!("annotations-{}",index)} class={border} onclick={ move |_|{ onclick.emit(index); }}>
        <img src={image_to_base64_image(&annotations.get_image())} width={32} height={32} />
        { format!(" annotations: {}",annotations.len()) }
        </div></li>}
    }
}

impl Component for ImagesList {
    type Message = Msg;
    type Properties = Props;

    fn create(_ctx: &Context<Self>) -> Self {
        Self
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::AddImage => {
                ctx.props().onaddimage.emit(());
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let on_add_image = ctx.link().callback(|_| Msg::AddImage);
        let current = ctx.props().current;
        html! {
            <div id="images-list" class="card" style="width: 18rem;">
            <ul class="list-group list-group-flush">
            <li class="list-group-item">
                <button type="button" class="btn btn-success" onclick={on_add_image}>
                    { "Add Image" }
                </button>
            </li>
            {
                ctx.props().images.clone().iter().enumerate().map(|(i, annotations)| {
                    self.create_image_element(i, annotations, current, ctx.props().onimageselected.clone())
                }).collect::<Html>()
            }
            </ul>
          </div>
        }
    }
}
