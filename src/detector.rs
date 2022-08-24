use crate::bbox::BBox;
use crate::detection::{merge, nms_sort, Detection};
use crate::HogDetector;
use crate::Predictable;
use image::{DynamicImage, RgbImage, Rgba, SubImage};
use imageproc::drawing::{draw_cross_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use rusttype::{Font, Scale};

pub fn windows(image: &RgbImage, window_size: u32) -> (u32, u32, Vec<SubImage<&RgbImage>>) {
    let cols = 2 * (image.width() / window_size);
    let rows = 2 * (image.height() / window_size);
    let mut subimages = Vec::new();

    for y in 0..rows - 1 {
        for x in 0..cols - 1 {
            subimages.push(SubImage::new(
                image,
                x * (window_size / 2),
                y * (window_size / 2),
                window_size,
                window_size,
            ))
        }
    }
    (cols, rows, subimages)
}

pub fn detect_objects(
    cols: u32,
    rows: u32,
    predictions: Vec<u32>,
    window_size: u32,
) -> Vec<Detection> {
    let mut detections: Vec<Detection> = Vec::new();
    for y in 0..rows - 1 {
        for x in 0..cols - 1 {
            let i = (y * (cols - 1) + x) as usize;
            let class = predictions[i] as usize;
            if class > 0 {
                // add 0.1 to generate an overlap on contacting windows.
                let size = 0.01 + window_size as f32;
                let bbox = BBox {
                    x: (x * (window_size / 2)) as f32,
                    y: (y * (window_size / 2)) as f32,
                    w: size,
                    h: size,
                };
                detections.push(Detection {
                    class,
                    bbox,
                    confidence: 1.0,
                });
            }
        }
    }
    let detections = merge(detections);
    nms_sort(detections)
}

pub trait Detector {
    fn detect_objects(&self, image: &DynamicImage) -> Vec<Detection>;
    fn visualize_detections(&self, image: &DynamicImage) -> DynamicImage;
}

impl Detector for HogDetector {
    fn detect_objects(&self, image: &DynamicImage) -> Vec<Detection> {
        let window_size = 32;
        let image = image.to_rgb8();
        let (cols, rows, windows) = windows(&image, window_size);
        let predictions: Vec<u32> = windows
            .iter()
            .map(|window| self.predict(&window.to_image()))
            .collect();
        detect_objects(cols, rows, predictions, window_size)
    }

    fn visualize_detections(&self, image: &DynamicImage) -> DynamicImage {
        let window_size = 32;
        let detections = self.detect_objects(image);

        let mut img_copy = image.to_rgba8();
        for detection in detections.iter() {
            let color = Rgba([125u8, 255u8, 0u8, 0u8]);
            draw_cross_mut(
                &mut img_copy,
                Rgba([255u8, 0u8, 0u8, 0u8]),
                detection.bbox.x as i32,
                detection.bbox.y as i32,
            );
            draw_hollow_rect_mut(
                &mut img_copy,
                Rect::at(
                    (detection.bbox.x as u32) as i32,
                    (detection.bbox.y as u32) as i32,
                )
                .of_size(window_size, window_size),
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainable::Trainable;
    use crate::DataSet;

    #[test]
    fn test_detect() {
        let mut model = HogDetector::default();

        let mut dataset = DataSet::new(
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

        let mut dataset = DataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            32,
        );
        dataset.load(true);

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
        let webcam10 = image::open("res/training/webcam10.jpg").unwrap();
        model
            .visualize_detections(&webcam10)
            .save("out/test_visualize_detections_5.png")
            .unwrap();
    }
}
