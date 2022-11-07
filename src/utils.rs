use image::{
    imageops, imageops::resize, imageops::FilterType, DynamicImage, Rgb, RgbImage, SubImage,
};
use imageproc::geometric_transformations::{rotate_about_center, warp, Interpolation, Projection};
use rand::prelude::ThreadRng;
use rand::Rng;

type X = u32;
type Y = u32;
type Window<'a> = (X, Y, RgbImage);

pub fn pyramid(image: &RgbImage, scale: f32, step_size: usize, window_size: u32) -> Vec<Window> {
    let width = image.width() as f32 / scale;
    let height = image.height() as f32 / scale;
    let image = resize(
        &DynamicImage::ImageRgb8(image.clone()),
        width as u32,
        height as u32,
        FilterType::Nearest,
    );
    let image = DynamicImage::ImageRgba8(image).to_rgb8();
    let windows = sliding_window(&image, step_size, window_size);
    windows
        .into_iter()
        .map(|(x, y, window)| ((x as f32 * scale) as u32, (y as f32 * scale) as u32, window))
        .collect()
}

pub fn sliding_window(image: &RgbImage, step_size: usize, window_size: u32) -> Vec<Window> {
    let mut windows = Vec::new();

    for y in (0..image.height() - window_size as u32).step_by(step_size) {
        for x in (0..image.width() - window_size as u32).step_by(step_size) {
            windows.push((
                x,
                y,
                SubImage::new(image, x, y, window_size, window_size).to_image(),
            ))
        }
    }
    windows
}

pub fn rotated_frames(frame: &RgbImage) -> impl Iterator<Item = RgbImage> + '_ {
    [
        0.02, -0.02, 0.05, -0.05, 0.07, -0.07, 0.09, -0.09, 1.1, -1.1, 1.3, -1.3, 1.5, -1.5, 2.0,
        -2.0,
    ]
    .iter()
    .map(|rad| rotate_about_center(frame, *rad, Interpolation::Nearest, Rgb([0, 0, 0])))
}

pub fn scaled_frames(frame: &RgbImage) -> impl Iterator<Item = RgbImage> + '_ {
    [0.8, 0.9, 1.1, 1.2].into_iter().map(|scalefactor| {
        let scale = Projection::scale(scalefactor, scalefactor);

        warp(frame, &scale, Interpolation::Nearest, Rgb([0, 0, 0]))
    })
}

pub fn window_crop(
    input_frame: &RgbImage,
    window_width: u32,
    window_height: u32,
    center: (u32, u32),
) -> RgbImage {
    imageops::crop(
        &mut input_frame.clone(),
        center
            .0
            .saturating_sub(window_width / 2)
            .min(input_frame.width() - window_width),
        center
            .1
            .saturating_sub(window_height / 2)
            .min(input_frame.height() - window_height),
        window_width,
        window_height,
    )
    .to_image()
}

pub fn generate_random_subimages(
    image: &RgbImage,
    count: usize,
    width: u32,
    height: u32,
) -> Vec<RgbImage> {
    let mut subimages = Vec::new();
    let mut rng: ThreadRng = rand::thread_rng();

    for _ in 0..count {
        let x = rng.gen_range(width / 2..=image.width());
        let y = rng.gen_range(height / 2..=image.height());
        subimages.push(window_crop(image, width, height, (x, y)));
    }
    subimages
}

#[cfg(test)]
mod tests {
    use super::*;
    use imageproc::utils::rgb_bench_image;

    #[test]
    fn test_sliding_window() {
        let image = RgbImage::new(10, 10);
        let step_size = 2;
        let window_size = 4;
        let windows = sliding_window(&image, step_size, window_size);
        assert_eq!(windows.len(), 9);
    }

    #[test]
    fn test_pyramid() {
        let image = RgbImage::new(16, 16);
        let step_size = 2;
        let window_size = 4;
        let windows = pyramid(&image, 2.0, step_size, window_size);
        assert_eq!(windows.len(), 4);
    }

    #[test]
    fn test_window_crop() {
        let image = rgb_bench_image(100, 100);
        let window = window_crop(&image, 8, 10, (20, 20));
        assert_eq!(8, window.width());
        assert_eq!(10, window.height());
    }

    #[test]
    fn test_generate_subimages() {
        let image = rgb_bench_image(100, 100);
        let subimages = generate_random_subimages(&image, 4, 8, 10);
        assert_eq!(4, subimages.len());
        assert_eq!(8, subimages[0].width());
        assert_eq!(10, subimages[0].height());
    }
}
