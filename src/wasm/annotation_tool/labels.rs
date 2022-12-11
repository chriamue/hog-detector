use super::Msg;
use yew::{html, Callback, Component, Context, Html, Properties};

pub struct Labels {}

#[derive(Clone, PartialEq, Properties)]
pub struct Props {
    pub label: String,
    pub labels: Vec<String>,
    pub onchange: Callback<String>,
}

impl Component for Labels {
    type Message = Msg;
    type Properties = Props;

    fn create(_ctx: &Context<Self>) -> Self {
        Self {}
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::LabelChanged(label) => {
                ctx.props().onchange.emit(label);
                true
            }
            _ => false,
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        html! {
            <div id="labels">
            <ul class="item-list">
        { ctx.props().labels.iter().map(|label| {

            let l = label.clone();
            if l == ctx.props().label {
                html!{<button type="button" class="btn btn-primary" onclick={
                    ctx.link().callback(move |_| Msg::LabelChanged(l.clone()))
                }>{ label.to_string() }</button>}
            } else {
                html!{<button type="button" class="btn btn-secondary" onclick={
                    ctx.link().callback(move |_| Msg::LabelChanged(l.clone()))
                }>{ label.to_string() }</button>}
            }

        }).collect::<Html>() }
            </ul>
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
        let label = "none".to_string();
        let labels: Vec<String> = vec!["none".to_string(), "some".to_string()];
        let onchange = Callback::default();
        let rendered = yew::LocalServerRenderer::<Labels>::with_props(
            Props {
                label, labels, onchange
            },
        )
        .render().await;
        assert!(rendered.contains("none"));
        assert!(rendered.contains("some"));
    }
}
