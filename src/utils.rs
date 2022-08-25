use image::{imageops::resize, imageops::FilterType, DynamicImage, RgbImage, SubImage};

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
    let windows = windows
        .into_iter()
        .map(|(x, y, window)| ((x as f32 * scale) as u32, (y as f32 * scale) as u32, window))
        .collect();
    windows
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
