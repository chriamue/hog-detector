use image::{DynamicImage, Rgba};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use object_detector_rust::prelude::BBox;
use rusttype::{Font, Scale};

use object_detector_rust::detection::{merge_overlapping_detections, Detection};
pub use object_detector_rust::detector::Detector;

/// detect_objects() takes in a vector of tuples containing x, y, and class values and a window size as parameters.
/// It creates a vector of Detection objects with the given x, y, class values and the window size plus 0.01
/// to generate an overlap on contacting windows.
/// The detections are then merged and sorted using non-maximum suppression (NMS).
/// The function returns the sorted vector of Detection objects.
pub fn detect_objects(predictions: Vec<(u32, u32, u32)>, window_size: u32) -> Vec<Detection> {
    let mut detections: Vec<Detection> = Vec::new();
    predictions.iter().for_each(|(x, y, class)| {
        if *class > 0 {
            // add 0.1 to generate an overlap on contacting windows.
            let size = (0.01 + window_size as f32) as u32;
            let bbox = BBox {
                x: *x as i32,
                y: *y as i32,
                width: size,
                height: size,
            };
            detections.push(Detection {
                class: *class,
                bbox,
                confidence: 1.0,
            });
        }
    });
    let detections = merge_overlapping_detections(&detections);
    detections
}

/// visualizes detections on given image
pub fn visualize_detections(image: &DynamicImage, detections: &Vec<Detection>) -> DynamicImage {
    let mut img_copy = image.to_rgba8();
    for detection in detections.iter() {
        let color = Rgba([125u8, 255u8, 0u8, 0u8]);
        draw_hollow_rect_mut(
            &mut img_copy,
            Rect::at(detection.bbox.x as i32, detection.bbox.y as i32)
                .of_size(detection.bbox.width, detection.bbox.height),
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;
    use object_detector_rust::tests::test_image;

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

        assert_eq!(&Rgb([0, 0, 0]), img.to_rgb8().get_pixel(8, 8));
        let img = visualize_detections(&img, &objects).to_rgb8();

        img.save("out/test_image2.png").unwrap();
        assert_eq!(&Rgb([125, 255, 0]), img.get_pixel(8, 8));
    }
}
