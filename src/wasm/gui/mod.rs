use yew::prelude::*;

pub mod header;
pub mod labels;

pub enum Msg {
    LabelChanged(String),
}

pub struct App {
    current_label: String,
}

impl Component for App {
    type Message = Msg;
    type Properties = ();

    fn create(_ctx: &Context<Self>) -> Self {
        Self {
            current_label: "none".to_string(),
        }
    }

    fn update(&mut self, _ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::LabelChanged(label) => {
                self.current_label = label;
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let label = self.current_label.clone();
        let onchange = ctx
            .link()
            .callback(|label: String| Msg::LabelChanged(label.to_string()));
        html! {
            <>
            <header::Header />
            { self.current_label.to_string() }
            <labels::Labels { onchange } {label} />
            </>
        }
    }
}
