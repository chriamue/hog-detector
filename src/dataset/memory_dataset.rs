use super::DataSet;
use crate::detection::Detection;
use image::{
    imageops::{crop, resize, FilterType},
    DynamicImage, RgbImage,
};

type Sample = (RgbImage, Vec<Detection>);

pub struct MemoryDataSet {
    data: Vec<Sample>,
    window_width: u32,
    window_height: u32,
}

impl Default for MemoryDataSet {
    fn default() -> Self {
        MemoryDataSet {
            data: Vec::new(),
            window_width: 32,
            window_height: 32,
        }
    }
}

impl MemoryDataSet {
    pub fn add(&mut self, sample: Sample) {
        self.data.push(sample);
    }
}

impl DataSet for MemoryDataSet {
    fn load(&mut self, _augment: bool) {}
    fn generate_random_annotations(&mut self, _count_each: usize) {}
    fn samples(&self) -> usize {
        self.data.len()
    }
    fn get(&self) -> (Vec<RgbImage>, Vec<u32>, Vec<RgbImage>, Vec<u32>) {
        let mut train_x = Vec::new();
        let mut train_y = Vec::new();
        let mut test_x = Vec::new();
        let mut test_y = Vec::new();

        for (img, annotations) in self.data.iter() {
            let mut img = img.clone();
            for annotation in annotations {
                let bbox = annotation.bbox;
                let window = crop(
                    &mut img,
                    bbox.x as u32,
                    bbox.y as u32,
                    bbox.w as u32,
                    bbox.h as u32,
                )
                .to_image();
                let scaled_window = resize(
                    &DynamicImage::ImageRgb8(window),
                    self.window_width,
                    self.window_height,
                    FilterType::Nearest,
                );
                let image = DynamicImage::ImageRgba8(scaled_window).to_rgb8();
                train_x.push(image.clone());
                train_y.push(annotation.class as u32);
                test_x.push(image);
                test_y.push(annotation.class as u32);
            }
        }
        (train_x, train_y, test_x, test_y)
    }
}

#[cfg(test)]
mod tests {
    use imageproc::utils::rgb_bench_image;

    use super::*;

    #[test]
    fn test_add_sample() {
        let mut dataset = MemoryDataSet::default();
        let sample = (rgb_bench_image(10, 10), vec![Detection::default()]);
        dataset.add(sample);
        assert_eq!(1, dataset.samples());
    }

    #[test]
    fn test_get_data() {
        let mut dataset = MemoryDataSet::default();
        let (width, height) = (10, 10);
        let mut detection = Detection::default();
        detection.bbox.w = 5.0;
        detection.bbox.h = 5.0;
        let sample = (rgb_bench_image(width, height), vec![detection]);
        dataset.add(sample);
        let (train_x, train_y, test_x, test_y) = dataset.get();
        assert_eq!(train_x.len(), train_y.len());
        assert_eq!(train_x.len(), test_x.len());
        assert_eq!(5, train_x[0].width());
        assert_eq!(5, train_x[0].height());
        assert_eq!(detection.class, test_y[0] as usize);
    }
}
