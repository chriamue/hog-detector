use image::{imageops, Rgb, RgbImage};
use imageproc::geometric_transformations::{rotate_about_center, warp, Interpolation, Projection};
use rand::prelude::ThreadRng;
use rand::Rng;

#[cfg(feature = "eyes")]
#[cfg(not(target_arch = "wasm32"))]
mod eyes_dataset;
#[cfg(not(target_arch = "wasm32"))]
mod folder_dataset;
mod memory_dataset;
#[cfg(feature = "mnist")]
#[cfg(not(target_arch = "wasm32"))]
mod mnist_dataset;

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

pub trait DataSet {
    fn load(&mut self, augment: bool);
    fn generate_random_annotations(&mut self, count_each: usize);
    fn samples(&self) -> usize;
    fn get(&self) -> (Vec<RgbImage>, Vec<u32>, Vec<RgbImage>, Vec<u32>);
}

#[cfg(feature = "eyes")]
#[cfg(not(target_arch = "wasm32"))]
pub use eyes_dataset::EyesDataSet;
#[cfg(not(target_arch = "wasm32"))]
pub use folder_dataset::FolderDataSet;
pub use memory_dataset::MemoryDataSet;
#[cfg(feature = "mnist")]
#[cfg(not(target_arch = "wasm32"))]
pub use mnist_dataset::MnistDataSet;

#[cfg(test)]
mod tests {
    use super::*;
    use imageproc::utils::rgb_bench_image;

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
