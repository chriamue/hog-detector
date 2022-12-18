use yew::{html, Component, Context, Html};

pub struct Header;

impl Component for Header {
    type Message = ();
    type Properties = ();

    fn create(_ctx: &Context<Self>) -> Self {
        Self
    }

    fn view(&self, _ctx: &Context<Self>) -> Html {
        html! {
            <div id="header" class="jumbotron mt-4 p-3 mb-5 bg-light rounded shadow">
                <h1>{"HOG Detector Annotation Tool"}</h1>
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
        let rendered = yew::LocalServerRenderer::<Header>::new().render().await;
        assert!(rendered.contains("<h1>HOG Detector Annotation Tool</h1>"));
    }
}
