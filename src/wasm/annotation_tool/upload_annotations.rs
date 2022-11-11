use js_sys::{ArrayBuffer, Uint8Array};
use wasm_bindgen::JsCast;
use web_sys::HtmlInputElement;
use yew::{html, Callback, Component, Context, Html, NodeRef, Properties};

pub struct UploadAnnotations {
    input_ref: NodeRef,
}

#[derive(Clone, PartialEq, Properties)]
pub struct Props {
    pub onchange: Callback<(String, Vec<u8>)>,
}

impl Component for UploadAnnotations {
    type Message = ();
    type Properties = Props;

    fn create(_ctx: &Context<Self>) -> Self {
        Self {
            input_ref: NodeRef::default(),
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let input_ref = self.input_ref.clone();
        let handle_change = {
            let input_ref = input_ref;
            let onchange = ctx.props().onchange.clone();
            Callback::from(move |_| {
                let input_element = input_ref
                    .get()
                    .unwrap()
                    .dyn_into::<HtmlInputElement>()
                    .unwrap();
                let onchange = onchange.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    if let Some(files) = input_element.files() {
                        if let Some(file) = files.item(0) {
                            let file_array_buffer_promise = file.array_buffer();
                            let file_array_buffer: ArrayBuffer =
                                wasm_bindgen_futures::JsFuture::from(file_array_buffer_promise)
                                    .await
                                    .expect("Should be able to get array buffer from uploaded file")
                                    .dyn_into()
                                    .unwrap();
                            let file_array_buffer = Uint8Array::new(file_array_buffer.as_ref());
                            let file_bytes = file_array_buffer.to_vec();
                            onchange.emit((file.name(), file_bytes));
                        }
                    }
                });
            })
        };

        html! {
            <div class="form-group">
                <label for="upload-image">{"Upload a labels text file"}</label>
                <input class="btn btn-success"
                    aria-description="Upload a text file"
                    type="file"
                    accept="text/*"
                    onchange={handle_change}
                    ref={self.input_ref.clone()}
                    disabled={false}
                />
            </div>
        }
    }
}
