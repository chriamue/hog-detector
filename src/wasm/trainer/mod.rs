use crate::wasm::trainer::dataset::create_dataset;

use super::HogDetectorJS;
use image_label_tool::prelude::LabelTool;
use wasm_bindgen_test::console_log;
use yew::prelude::*;

mod dataset;

#[derive(Debug)]
pub struct TrainerApp {}

pub enum Msg {
    Train,
    TrainWithHardNegativeSamples,
    SwitchBayesClassifier,
    SwitchRandomForestClassifier,
    SwitchCombinedClassifier,
    SwitchSVMClassifier,
}

#[derive(Clone, PartialEq, Properties)]
pub struct Props {
    pub detector: HogDetectorJS,
    pub label_tool: LabelTool,
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
                let dataset = create_dataset(&ctx.props().label_tool);
                ctx.props().detector.train(dataset);
                console_log!("training done");
                true
            }
            Msg::TrainWithHardNegativeSamples => {
                console_log!("training with hard negative samples started...");
                let dataset = create_dataset(&ctx.props().label_tool);
                ctx.props()
                    .detector
                    .train_with_hard_negative_samples(dataset);
                console_log!("training done");
                true
            }
            Msg::SwitchBayesClassifier => {
                ctx.props().detector.init_bayes_classifier();
                log::info!("Switched to Bayes Classifier");
                true
            }
            Msg::SwitchRandomForestClassifier => {
                ctx.props().detector.init_random_forest_classifier();
                log::info!("Switched to Random Forest Classifier");
                true
            }
            Msg::SwitchCombinedClassifier => {
                ctx.props().detector.init_combined_classifier();
                log::info!("Switched to Combined Classifier");
                true
            }
            Msg::SwitchSVMClassifier => {
                ctx.props().detector.init_svm_classifier();
                log::info!("Switched to SVM Classifier");
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
        let onclick_svm = ctx.link().callback(|_| Msg::SwitchSVMClassifier);
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
            <button type="button" class="btn btn-success" onclick={onclick_svm}>
            { "Switch to SVM Classifier" }
            </button>
            <button type="button" class="btn btn-success" onclick={onclick_combined}>
            { "Switch to Combined Classifier" }
            </button>
            </div>
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    async fn test_render() {
        let detector = HogDetectorJS::new();
        let label_tool = LabelTool::default();

        let rendered = yew::LocalServerRenderer::<TrainerApp>::with_props(Props {
            detector: detector,
            label_tool: label_tool,
        })
        .render()
        .await;
        assert!(rendered.contains("Train Detector"));
        assert!(rendered.contains("Train Detector with hard negative samples"));
        assert!(rendered.contains("Switch to Random Forest Classifier"));
        assert!(rendered.contains("Switch to SVM Classifier"));
        assert!(rendered.contains("Switch to Combined Classifier"));
    }
}
