use crate::bbox::BBox;
use crate::detection::{merge, nms_sort, Detection};
use crate::utils::{pyramid, sliding_window};
use crate::HogDetector;
use crate::Predictable;
use image::{DynamicImage, Rgba};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use rusttype::{Font, Scale};

pub fn detect_objects(predictions: Vec<(u32, u32, u32)>, window_size: u32) -> Vec<Detection> {
    let mut detections: Vec<Detection> = Vec::new();
    predictions.iter().for_each(|(x, y, class)| {
        if *class > 0 {
            // add 0.1 to generate an overlap on contacting windows.
            let size = 0.01 + window_size as f32;
            let bbox = BBox {
                x: *x as f32,
                y: *y as f32,
                w: size,
                h: size,
            };
            detections.push(Detection {
                class: *class as usize,
                bbox,
                confidence: 1.0,
            });
        }
    });
    let detections = merge(detections);
    nms_sort(detections)
}

pub fn visualize_detections(image: &DynamicImage, detections: &Vec<Detection>) -> DynamicImage {
    let mut img_copy = image.to_rgba8();
    for detection in detections.iter() {
        let color = Rgba([125u8, 255u8, 0u8, 0u8]);
        draw_hollow_rect_mut(
            &mut img_copy,
            Rect::at(detection.bbox.x as i32, detection.bbox.y as i32)
                .of_size(detection.bbox.w as u32, detection.bbox.h as u32),
            color,
        );

        let font_data = include_bytes!("../res/Arial.ttf");
        let font = Font::try_from_bytes(font_data as &[u8]).unwrap();

        const FONT_SCALE: f32 = 10.0;

        draw_text_mut(
            &mut img_copy,
            Rgba([125u8, 255u8, 0u8, 0u8]),
            detection.bbox.x as i32,
            detection.bbox.y as i32,
            Scale::uniform(FONT_SCALE),
            &font,
            &format!("{}", detection.class),
        );
    }
    DynamicImage::ImageRgba8(img_copy)
}

pub trait Detector {
    fn detect_objects(&self, image: &DynamicImage) -> Vec<Detection>;
    fn visualize_detections(&self, image: &DynamicImage) -> DynamicImage;
}

impl Detector for HogDetector {
    fn detect_objects(&self, image: &DynamicImage) -> Vec<Detection> {
        let step_size = 8;
        let window_size = 32;
        let image = image.to_rgb8();
        let mut windows = sliding_window(&image, step_size, window_size);
        windows.extend(pyramid(&image, 1.3, step_size, window_size));
        windows.extend(pyramid(&image, 1.5, step_size, window_size));

        let predictions: Vec<(u32, u32, u32)> = windows
            .iter()
            .map(|(x, y, window)| (*x, *y, self.predict(window)))
            .collect();
        detect_objects(predictions, window_size)
    }

    fn visualize_detections(&self, image: &DynamicImage) -> DynamicImage {
        let detections = self.detect_objects(image);
        visualize_detections(image, &detections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{DataSet, FolderDataSet};
    use crate::trainable::Trainable;

    #[test]
    fn test_detect() {
        let mut model = HogDetector::default();

        let mut dataset = FolderDataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            32,
        );
        dataset.load(false);

        model.train_class(&dataset, 5);
        assert!(model.svc.is_some());
        let webcam01 = image::open("res/training/webcam01.jpg").unwrap();
        let detections = model.detect_objects(&webcam01);
        println!("{:?}", detections);
        assert!(detections.len() > 0);
    }

    #[test]
    fn test_visualize_detections() {
        let mut model = HogDetector::default();

        let mut dataset = FolderDataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            32,
        );
        dataset.load(false);
        dataset.generate_random_annotations(10);

        model.train_class(&dataset, 1);
        assert!(model.svc.is_some());
        let webcam01 = image::open("res/training/webcam01.jpg").unwrap();
        model
            .visualize_detections(&webcam01)
            .save("out/test_visualize_detections_1.png")
            .unwrap();

        model.train_class(&dataset, 2);
        let webcam06 = image::open("res/training/webcam06.jpg").unwrap();
        model
            .visualize_detections(&webcam06)
            .save("out/test_visualize_detections_2.png")
            .unwrap();
        model.train_class(&dataset, 5);
        dataset.generate_hard_negative_samples(&model, 5);
        let webcam10 = image::open("res/training/webcam01.jpg").unwrap();
        model
            .visualize_detections(&webcam10)
            .save("out/test_visualize_detections_5_01.png")
            .unwrap();
        let webcam10 = image::open("res/training/webcam06.jpg").unwrap();
        model
            .visualize_detections(&webcam10)
            .save("out/test_visualize_detections_5_06.png")
            .unwrap();
        let webcam10 = image::open("res/training/webcam10.jpg").unwrap();
        model
            .visualize_detections(&webcam10)
            .save("out/test_visualize_detections_5_10.png")
            .unwrap();
    }
}
