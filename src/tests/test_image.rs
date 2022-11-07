use image::{Rgb, RgbImage};
use imageproc::{
    drawing::{draw_filled_circle_mut, draw_filled_ellipse_mut, draw_filled_rect_mut},
    rect::Rect,
    utils::rgb_bench_image,
};

pub fn test_image() -> RgbImage {
    let mut img = rgb_bench_image(100, 100);
    draw_filled_circle_mut(&mut img, (25, 25), 25, Rgb([255, 0, 0]));
    draw_filled_rect_mut(&mut img, Rect::at(25, 75).of_size(10, 10), Rgb([0, 0, 255]));
    draw_filled_ellipse_mut(&mut img, (75, 75), 25, 10, Rgb([0, 255, 0]));
    img
}

#[test]
fn test_image_size() {
    let img = test_image();
    assert_eq!(100, img.width());
    assert_eq!(100, img.height());
}

#[test]
fn test_pixel_color() {
    let img = test_image();
    assert_eq!(&Rgb([255, 0, 0]), img.get_pixel(1, 25));
    assert_eq!(&Rgb([0, 255, 0]), img.get_pixel(55, 80));
    assert_eq!(&Rgb([0, 0, 255]), img.get_pixel(30, 80));
}
