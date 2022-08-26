use crate::Detector;
use crate::HogDetector;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct HogDetectorJS {
    hog: HogDetector,
}

#[wasm_bindgen]
impl HogDetectorJS {
    #[wasm_bindgen(constructor)]
    pub fn new() -> HogDetectorJS {
        use console_error_panic_hook;
        console_error_panic_hook::set_once();

        let hog = {
            let model = include_str!("../res/eyes_model.json");
            serde_json::from_str::<HogDetector>(&model).unwrap()
        };

        HogDetectorJS { hog }
    }

    #[wasm_bindgen]
    pub fn next(&mut self, img_data: &[u8]) -> Vec<u8> {
        let mut img =
            image::load_from_memory_with_format(img_data, image::ImageFormat::Png).unwrap();

        img = self.hog.visualize_detections(&mut img);

        let mut image_data: Vec<u8> = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut image_data),
            image::ImageFormat::Png,
        )
        .unwrap();
        image_data
    }
}
