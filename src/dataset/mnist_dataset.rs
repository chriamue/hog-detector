use crate::dataset::DataSet;
use image::{imageops::resize, imageops::FilterType, DynamicImage, ImageBuffer, RgbImage};
use mnist::*;
use ndarray::prelude::*;

/// the well known mnist dataset
pub struct MnistDataSet {
    mnist: Option<Mnist>,
    data_len: usize,
}

impl Default for MnistDataSet {
    fn default() -> Self {
        MnistDataSet {
            mnist: None,
            data_len: 100,
        }
    }
}

impl MnistDataSet {}

fn bw_ndarray2_to_rgb_image(arr: Array2<f32>) -> RgbImage {
    assert!(arr.is_standard_layout());

    let (width, height) = (arr.ncols(), arr.ncols());
    let mut img: RgbImage = ImageBuffer::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            let val = (arr[[y, x]] * 255.) as u8;
            img.put_pixel(x as u32, y as u32, image::Rgb([val, val, val]))
        }
    }
    let img = resize(&DynamicImage::ImageRgb8(img), 32, 32, FilterType::Gaussian);
    DynamicImage::ImageRgba8(img).to_rgb8()
}

impl DataSet for MnistDataSet {
    fn load(&mut self, _augment: bool) {
        self.mnist = Some(
            MnistBuilder::new()
                .label_format_digit()
                .training_set_length(self.data_len as u32)
                .validation_set_length(0)
                .test_set_length(self.data_len as u32)
                .base_path("out/mnist/")
                .download_and_extract()
                .finalize(),
        );
    }

    fn generate_random_annotations(&mut self, _count_each: usize) {}

    fn get(&self) -> (Vec<RgbImage>, Vec<u32>, Vec<RgbImage>, Vec<u32>) {
        let mut train_x = Vec::new();
        let mut train_y = Vec::new();

        let mut test_x = Vec::new();
        let mut test_y = Vec::new();

        let Mnist {
            trn_img,
            trn_lbl,
            tst_img,
            tst_lbl,
            ..
        } = self.mnist.as_ref().unwrap();

        let train_data = Array3::from_shape_vec((self.data_len, 28, 28), trn_img.to_vec())
            .expect("Error converting images to Array3 struct")
            .mapv(|x| x as f32 / 256.);

        let test_data = Array3::from_shape_vec((self.data_len, 28, 28), tst_img.to_vec())
            .expect("Error converting images to Array3 struct")
            .mapv(|x| x as f32 / 256.);

        for item_num in 0..self.data_len {
            let image = bw_ndarray2_to_rgb_image(train_data.slice(s![item_num, .., ..]).to_owned());
            train_x.push(image);
            train_y.push(trn_lbl[item_num] as u32);

            let image = bw_ndarray2_to_rgb_image(test_data.slice(s![item_num, .., ..]).to_owned());
            test_x.push(image);
            test_y.push(tst_lbl[item_num] as u32);
        }

        (train_x, train_y, test_x, test_y)
    }

    fn samples(&self) -> usize {
        self.data_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let mut dataset = MnistDataSet::default();
        dataset.load(false);
        assert_eq!(dataset.samples(), 100);
        assert_eq!(dataset.get().0.len(), dataset.samples());
        assert_eq!(dataset.get().2.len(), dataset.samples());
    }
}
