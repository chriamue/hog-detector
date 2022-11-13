use crate::bbox::BBox;
use crate::detection::{merge, nms_sort, Detection};
use crate::utils::{pyramid, sliding_window};
use crate::HogDetector;
use crate::Predictable;
use image::{DynamicImage, Rgba};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use rusttype::{Font, Scale};

/// converts list of (x, y, class) into detections
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

/// visualizes detections on given image
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

/// detector trait
pub trait Detector {
    /// detects objects im given image
    fn detect_objects(&self, image: &DynamicImage) -> Vec<Detection>;
    /// visualize detections on image
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
    use crate::dataset::{DataSet, MemoryDataSet};
    use crate::tests::test_image;
    use crate::Trainable;
    use image::Rgb;

    #[test]
    fn test_detect_objects() {
        let detections = vec![(25, 25, 1), (30, 30, 1), (40, 40, 2)];
        let objects = detect_objects(detections, 16);
        assert_eq!(2, objects.len());
    }

    #[test]
    fn test_visualize_detections() {
        let detections = vec![(8, 8, 1), (40, 40, 2)];
        let objects = detect_objects(detections, 16);
        let img = test_image();

        assert_eq!(&Rgb([0, 0, 0]), img.get_pixel(8, 8));
        let img = visualize_detections(&DynamicImage::ImageRgb8(img), &objects).to_rgb8();

        img.save("out/test_image2.png").unwrap();
        assert_eq!(&Rgb([125, 255, 0]), img.get_pixel(8, 8));
    }

    #[test]
    fn test_detector() {
        let img = DynamicImage::ImageRgb8(test_image());
        let mut dataset = MemoryDataSet::new_test();
        dataset.load();
        let mut detector = HogDetector::default();
        detector.train_class(&dataset, 1);
        let detections = detector.detect_objects(&img);
        assert_eq!(1, detections.len());
        assert!(detections[0].bbox.x < 75.0);
        assert!(detections[0].bbox.x > 25.0);
        assert!(detections[0].bbox.y < 25.0);
        assert!(detections[0].bbox.y >= 0.0);
        let visualization = detector.visualize_detections(&img).to_rgb8();
        assert_eq!(&Rgb([125, 255, 0]), visualization.get_pixel(55, 0));
        assert_eq!(&Rgb([125, 255, 0]), visualization.get_pixel(75, 0));
    }
}
