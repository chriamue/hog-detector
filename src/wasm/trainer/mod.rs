use super::annotated_images_js::AnnotatedImagesJS;
use super::HogDetectorJS;
use wasm_bindgen_test::console_log;
use yew::prelude::*;

pub struct TrainerApp {}

pub enum Msg {
    Train,
    TrainWithHardNegativeSamples,
    SwitchBayesClassifier,
    SwitchRandomForestClassifier,
    SwitchCombinedClassifier,
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
                ctx.props().detector.train(dataset);
                console_log!("training done");
                true
            }
            Msg::TrainWithHardNegativeSamples => {
                console_log!("training with hard negative samples started...");
                let dataset = ctx.props().images.create_dataset();
                ctx.props()
                    .detector
                    .train_with_hard_negative_samples(dataset);
                console_log!("training done");
                true
            }
            Msg::SwitchBayesClassifier => {
                ctx.props().detector.init_bayes_classifier();
                true
            }
            Msg::SwitchRandomForestClassifier => {
                ctx.props().detector.init_random_forest_classifier();
                true
            }
            Msg::SwitchCombinedClassifier => {
                ctx.props().detector.init_combined_classifier();
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let onclick_train = ctx.link().callback(|_| Msg::Train);
        let onclick_train_with_hard_negative_samples =
            ctx.link().callback(|_| Msg::TrainWithHardNegativeSamples);
        let onclick_bayes = ctx.link().callback(|_| Msg::SwitchBayesClassifier);
        let onclick_random_forest = ctx.link().callback(|_| Msg::SwitchRandomForestClassifier);
        let onclick_combined = ctx.link().callback(|_| Msg::SwitchCombinedClassifier);
        html! {
            <div id="train-classifier-buttons">
            <button type="button" class="btn btn-success" onclick={onclick_train}>
            { "Train Detector" }
            </button>
            <button type="button" class="btn btn-success" onclick={onclick_train_with_hard_negative_samples}>
            { "Train Detector with hard negative samples" }
            </button>
            <button type="button" class="btn btn-success" onclick={onclick_bayes}>
            { "Switch to Naive Bayes Classifier" }
            </button>
            <button type="button" class="btn btn-success" onclick={onclick_random_forest}>
            { "Switch to Random Forest Classifier" }
            </button>
            <button type="button" class="btn btn-success" onclick={onclick_combined}>
            { "Switch to Combined Classifier" }
            </button>
            </div>
        }
    }
}
