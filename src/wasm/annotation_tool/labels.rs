use super::Msg;
use yew::{html, Callback, Component, Context, Html, Properties};

pub struct Labels {
    labels: Vec<String>,
}

#[derive(Clone, PartialEq, Properties)]
pub struct Props {
    pub label: String,
    pub onchange: Callback<String>,
}

impl Component for Labels {
    type Message = Msg;
    type Properties = Props;

    fn create(_ctx: &Context<Self>) -> Self {
        Self {
            labels: include_str!("../../../res/labels.txt")
                .split('\n')
                .map(|s| s.to_string())
                .collect(),
        }
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
            <ul class="item-list">
        { self.labels.iter().map(|label| {

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
        }
    }
}
