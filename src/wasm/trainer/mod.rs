use super::annotated_images_js::AnnotatedImagesJS;
use super::HogDetectorJS;
use wasm_bindgen_test::console_log;
use yew::prelude::*;

pub struct TrainerApp {}

pub enum Msg {
    Train,
}

#[derive(Clone, PartialEq, Properties)]
pub struct Props {
    pub detector: HogDetectorJS,
    pub images: AnnotatedImagesJS,
}

impl Component for TrainerApp {
    type Message = Msg;
    type Properties = Props;

    fn create(_ctx: &Context<Self>) -> Self {
        Self {}
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::Train => {
                console_log!("training started...");
                let dataset = ctx.props().images.create_dataset();
                ctx.props().detector.train(&dataset);
                console_log!("training done");
                console_log!("{}", ctx.props().images);
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let onclick = ctx.link().callback(|_| Msg::Train);
        html! {
            <>
                <button type="button" class="btn btn-success" {onclick}>
                    { "Train Detector" }
                </button>
            </>
        }
    }
}
