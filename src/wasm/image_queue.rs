use std::{collections::VecDeque, sync::Mutex};

use wasm_bindgen_futures::js_sys::Math::random;
use web_sys::ImageData;

pub struct ImageQueue {
    id: usize,
    queue: Mutex<VecDeque<ImageData>>,
    max_size: usize,
}

impl PartialEq for ImageQueue {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl ImageQueue {
    pub fn new(max_size: usize) -> Self {
        Self {
            id: (random() * 1000.0) as usize,
            queue: Mutex::new(VecDeque::new()),
            max_size,
        }
    }

    pub fn new_with_id(id: usize, max_size: usize) -> Self {
        Self {
            id,
            queue: Mutex::new(VecDeque::new()),
            max_size,
        }
    }

    pub fn push(&self, image_data: ImageData) -> Result<(), String> {
        if let Ok(mut queue) = self.queue.try_lock() {
            if queue.len() >= self.max_size {
                queue.pop_front();
            }
            queue.push_back(image_data);
            Ok(())
        } else {
            Err("Failed to acquire lock".into())
        }
    }

    pub fn pop(&self) -> Option<ImageData> {
        if let Ok(mut queue) = self.queue.try_lock() {
            queue.pop_front()
        } else {
            None
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }
}

#[cfg(test)]
mod tests {

    use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};
    wasm_bindgen_test_configure!(run_in_browser);

    use super::*;

    #[wasm_bindgen_test]
    fn test_image_queue() {
        let image_queue = ImageQueue::new(3);
        let image_data = ImageData::new_with_sw(32, 32).unwrap();
        image_queue.push(image_data.clone()).unwrap();
        assert_eq!(image_queue.pop(), Some(image_data));
    }
}
