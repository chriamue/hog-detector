use super::HogDetectorJS;
use yew::prelude::*;

pub struct TrainerApp {}

#[derive(Clone, PartialEq, Properties)]
pub struct Props {
    pub detector: HogDetectorJS,
}

impl Component for TrainerApp {
    type Message = ();
    type Properties = Props;

    fn create(_ctx: &Context<Self>) -> Self {
        Self {}
    }

    fn view(&self, _ctx: &Context<Self>) -> Html {
        html! {
            <>
                <button type="button" class="btn btn-success">
                    { "Train Detector" }
                </button>
            </>
        }
    }
}
