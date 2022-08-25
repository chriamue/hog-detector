use image::{RgbImage, SubImage};

type X = u32;
type Y = u32;
type Window<'a> = (X, Y, SubImage<&'a RgbImage>);

pub fn sliding_window(image: &RgbImage, step_size: usize, window_size: u32) -> Vec<Window> {
    let mut windows = Vec::new();

    for y in (0..image.height() - step_size as u32).step_by(step_size) {
        for x in (0..image.width() - step_size as u32).step_by(step_size) {
            windows.push((x, y, SubImage::new(image, x, y, window_size, window_size)))
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
        assert_eq!(windows.len(), 16);
    }
}
