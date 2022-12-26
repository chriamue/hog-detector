use std::io::Cursor;

use image::imageops::blur;
use image::{imageops, imageops::resize, imageops::FilterType, DynamicImage, Rgb};
use image::{GenericImageView, ImageOutputFormat};
use imageproc::corners::corners_fast9;
use imageproc::geometric_transformations::{rotate_about_center, warp, Interpolation, Projection};
use rand::prelude::ThreadRng;
use rand::Rng;

type X = u32;
type Y = u32;
type Window<'a> = (X, Y, DynamicImage);

/// This function calculates a pyramid of windows from an image.
/// The scale, step size, and window size are all arguments that can be passed in.
/// The image is resized to the width and height divided by the scale.
/// A sliding window is then created with the step size and window size arguments.
/// The windows are then mapped to their original coordinates multiplied by the scale
/// and collected into a vector of Windows.
pub fn pyramid(
    image: &DynamicImage,
    scale: f32,
    step_size: usize,
    window_size: u32,
) -> Vec<Window> {
    let width = image.width() as f32 / scale;
    let height = image.height() as f32 / scale;
    let image = resize(image, width as u32, height as u32, FilterType::Nearest);
    let image = DynamicImage::from(image);
    let windows = sliding_window(&image, step_size, window_size);
    windows
        .into_iter()
        .map(|(x, y, window)| ((x as f32 * scale) as u32, (y as f32 * scale) as u32, window))
        .collect()
}

/// This function calculates a sliding window based on an image, a step size and a window size.
/// It returns a vector of windows, which are composed of the x and y coordinates of the top left corner
/// of the window, as well as the DynamicImage associated with that window.
pub fn sliding_window(image: &DynamicImage, step_size: usize, window_size: u32) -> Vec<Window> {
    let mut windows = Vec::new();

    for y in (0..image.height() - window_size as u32).step_by(step_size) {
        for x in (0..image.width() - window_size as u32).step_by(step_size) {
            windows.push((
                x,
                y,
                DynamicImage::from(image.view(x, y, window_size, window_size).to_image()),
            ))
        }
    }
    windows
}

/// This function calculates keypoints based on windows from an image.
/// It takes in a DynamicImage, the number of keypoints to be calculated (count),
/// and the size of each window (window_size).
/// It first creates an empty vector to store the windows.
/// It then gets the width and height of the image, and calculates the center of each window.
/// The image is then blurred with a blur factor of 2.5,
/// and corners_fast9 is used to calculate keypoints from the blurred image.
/// The keypoints are sorted by score, and then iterated over until count is reached.
/// For each keypoint, its x and y coordinates are found along with its window size,
/// and pushed into the vector of windows. Finally, this vector is returned.
pub fn keypoint_windows(image: &DynamicImage, count: usize, window_size: u32) -> Vec<Window> {
    let mut windows = Vec::new();
    let (width, height) = image.dimensions();
    let center = window_size / 2;

    let blurred = blur(&image.to_luma8(), 2.5);
    let mut keypoints = corners_fast9(&blurred, 10);
    // sort keypoints by score
    keypoints.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

    // iterate over count keypoints
    for keypoint in keypoints.iter().take(count) {
        let x = keypoint.x + center;
        let y = keypoint.y + center;
        windows.push((
            x,
            y,
            DynamicImage::from(
                image
                    .view(
                        x.max(0).min(width - window_size),
                        y.max(0).min(height - window_size),
                        window_size,
                        window_size,
                    )
                    .to_image(),
            ),
        ))
    }
    windows
}

/// rotated_frames() takes in a DynamicImage and returns an iterator over the rotated windows
/// of the given image. It uses rotate_about_center() to rotate the image by a given radian angle,
/// and uses Nearest interpolation with a black background.
/// The radian angles used are 0.02, -0.02, 0.05, -0.05, 0.07, -0.07, 0.09, -0.09, 1.1, -1.1, 1.3, -1.3, 1.5, -1.5 and 2.0,-2.0
pub fn rotated_frames(frame: &DynamicImage) -> impl Iterator<Item = DynamicImage> + '_ {
    [
        0.02, -0.02, 0.05, -0.05, 0.07, -0.07, 0.09, -0.09, 1.1, -1.1, 1.3, -1.3, 1.5, -1.5, 2.0,
        -2.0,
    ]
    .iter()
    .map(|rad| {
        DynamicImage::from(rotate_about_center(
            &frame.to_rgb8(),
            *rad,
            Interpolation::Nearest,
            Rgb([0, 0, 0]),
        ))
    })
}

/// scaled_frames() takes in a DynamicImage and returns an iterator over scaled frames of the given image.
/// The iterator contains four frames, with scaling factors of 0.8, 0.9, 1.1, and 1.2 respectively.
/// The warp() function is used to scale the image using nearest neighbor interpolation
/// and a black background color (Rgb([0, 0, 0])).
pub fn scaled_frames(frame: &DynamicImage) -> impl Iterator<Item = DynamicImage> + '_ {
    [0.8, 0.9, 1.1, 1.2].into_iter().map(|scalefactor| {
        let scale = Projection::scale(scalefactor, scalefactor);
        DynamicImage::from(warp(
            &frame.to_rgb8(),
            &scale,
            Interpolation::Nearest,
            Rgb([0, 0, 0]),
        ))
    })
}

/// window_crop() takes in an image, a window width and height, and a center point (x, y) as parameters.
/// It then crops the image to the size of the window width and height,
/// with the center point being the center of the cropped image.
/// It returns a DynamicImage type with the cropped image.
pub fn window_crop(
    input_frame: &DynamicImage,
    window_width: u32,
    window_height: u32,
    center: (u32, u32),
) -> DynamicImage {
    let cropped = imageops::crop(
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
    .to_image();
    DynamicImage::ImageRgba8(cropped)
}

/// generate_random_subimages() takes in an image, a count,
/// and two dimensions (width and height) as parameters and returns a vector of DynamicImages.
/// It creates a ThreadRng to generate random numbers between the width/2 and the image's width,
/// and between the height/2 and the image's height. It then uses window_crop()
/// to crop the image at those coordinates with the given dimensions,
/// and pushes it into the vector of DynamicImages.
pub fn generate_random_subimages(
    image: &DynamicImage,
    count: usize,
    width: u32,
    height: u32,
) -> Vec<DynamicImage> {
    let mut subimages = Vec::new();
    let mut rng: ThreadRng = rand::thread_rng();

    for _ in 0..count {
        let x = rng.gen_range(width / 2..=image.width());
        let y = rng.gen_range(height / 2..=image.height());
        subimages.push(window_crop(image, width, height, (x, y)));
    }
    subimages
}

/// converts dynamic image to base64 encoded png image
pub fn image_to_base64_image(img: &DynamicImage) -> String {
    let mut image_data: Vec<u8> = Vec::new();
    img.write_to(&mut Cursor::new(&mut image_data), ImageOutputFormat::Png)
        .unwrap();
    let res_base64 = base64::encode(image_data);
    format!("data:image/png;base64,{}", res_base64)
}

/// decodes base64 encoded image to dynamic image
pub fn base64_image_to_image(b64img: &str) -> DynamicImage {
    let b64img = b64img.replace("data:image/png;base64,", "");
    let data = base64::decode(b64img).unwrap();
    image::load_from_memory(&data).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;
    use imageproc::utils::rgb_bench_image;

    #[test]
    fn test_sliding_window() {
        let image = DynamicImage::ImageRgb8(RgbImage::new(10, 10));
        let step_size = 2;
        let window_size = 4;
        let windows = sliding_window(&image, step_size, window_size);
        assert_eq!(windows.len(), 9);
    }

    #[test]
    fn test_pyramid() {
        let image = DynamicImage::ImageRgb8(RgbImage::new(16, 16));
        let step_size = 2;
        let window_size = 4;
        let windows = pyramid(&image, 2.0, step_size, window_size);
        assert_eq!(windows.len(), 4);
    }

    #[test]
    fn test_window_crop() {
        let image = DynamicImage::ImageRgb8(rgb_bench_image(100, 100));
        let window = window_crop(&image, 8, 10, (20, 20));
        assert_eq!(8, window.width());
        assert_eq!(10, window.height());
    }

    #[test]
    fn test_keypoint_windows() {
        let image = DynamicImage::ImageRgb8(rgb_bench_image(100, 100));
        let count = 2;
        let window_size = 4;
        let windows = keypoint_windows(&image, count, window_size);
        assert!(windows.len() <= count);
    }

    #[test]
    fn test_generate_subimages() {
        let image = DynamicImage::ImageRgb8(rgb_bench_image(100, 100));
        let subimages = generate_random_subimages(&image, 4, 8, 10);
        assert_eq!(4, subimages.len());
        assert_eq!(8, subimages[0].width());
        assert_eq!(10, subimages[0].height());
    }

    #[test]
    fn test_image_base64_encoding_and_decoding() {
        let image = DynamicImage::ImageRgb8(rgb_bench_image(100, 100));
        let encoded = image_to_base64_image(&image);
        assert!(encoded.starts_with("data:image/png;base64"));
        let decoded = base64_image_to_image(&encoded).to_rgb8();
        assert_eq!(image.width(), decoded.width());
        assert_eq!(image.height(), decoded.height());
    }
}
