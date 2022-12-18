use image_label_tool::prelude::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
/// init label tool and start app on given root html element
pub fn init_image_label_tool(root: web_sys::Element) -> LabelTool {
    image_label_tool::init_label_tool(root)
}

fn main() {
    eprintln!("wasm only example: run wasm-pack build --example web");
}
