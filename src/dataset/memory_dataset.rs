use super::DataSet;
use crate::utils::generate_random_subimages;
use crate::Annotation;
use image::{
    imageops::{crop, resize, FilterType},
    DynamicImage,
};

type Sample = (DynamicImage, Vec<Annotation>);

/// Memory only dataset
pub struct MemoryDataSet {
    samples: Vec<Sample>,
    x: Vec<DynamicImage>,
    y: Vec<u32>,
    window_width: u32,
    window_height: u32,
}

impl Default for MemoryDataSet {
    fn default() -> Self {
        MemoryDataSet {
            samples: Vec::new(),
            x: Vec::new(),
            y: Vec::new(),
            window_width: 32,
            window_height: 32,
        }
    }
}

impl MemoryDataSet {
    /// adds sample to dataset
    pub fn add(&mut self, sample: Sample) {
        self.samples.push(sample);
    }

    #[cfg(test)]
    pub fn new_test() -> Self {
        use crate::{bbox::BBox, tests::test_image};

        let mut dataset = MemoryDataSet::default();
        let sample = (
            test_image(),
            vec![
                (
                    BBox {
                        x: 0.0,
                        y: 0.0,
                        w: 50.0,
                        h: 50.0,
                    },
                    0,
                ),
                (
                    BBox {
                        x: 50.0,
                        y: 0.0,
                        w: 50.0,
                        h: 50.0,
                    },
                    1,
                ),
                (
                    BBox {
                        x: 0.0,
                        y: 50.0,
                        w: 50.0,
                        h: 50.0,
                    },
                    2,
                ),
                (
                    BBox {
                        x: 50.0,
                        y: 50.0,
                        w: 50.0,
                        h: 50.0,
                    },
                    3,
                ),
            ],
        );
        dataset.add(sample);
        dataset
    }
}

impl DataSet for MemoryDataSet {
    fn load(&mut self) {
        self.x.clear();
        self.y.clear();
        for (img, annotations) in self.samples.iter() {
            let mut img = img.clone();
            for annotation in annotations {
                let (bbox, class) = annotation;
                let window = crop(
                    &mut img,
                    bbox.x as u32,
                    bbox.y as u32,
                    bbox.w as u32,
                    bbox.h as u32,
                )
                .to_image();
                let scaled_window = resize(
                    &window,
                    self.window_width,
                    self.window_height,
                    FilterType::Nearest,
                );
                let image = DynamicImage::ImageRgba8(scaled_window);
                self.x.push(image);
                self.y.push(*class);
            }
        }
    }

    fn generate_random_annotations(&mut self, count_each: usize) {
        for sample in self.samples.iter() {
            let subimages = generate_random_subimages(
                &sample.0,
                count_each,
                self.window_width,
                self.window_height,
            );
            for subimage in subimages {
                self.x.push(subimage);
                self.y.push(0);
            }
        }
    }

    fn samples(&self) -> usize {
        self.y.len()
    }
    fn get(&self) -> (Vec<DynamicImage>, Vec<u32>, Vec<DynamicImage>, Vec<u32>) {
        let mut train_x = Vec::new();
        let mut train_y = Vec::new();
        let mut test_x = Vec::new();
        let mut test_y = Vec::new();

        for x in self.x.iter() {
            train_x.push(x.clone());
            test_x.push(x.clone());
        }
        for y in self.y.iter() {
            train_y.push(*y);
            test_y.push(*y);
        }
        (train_x, train_y, test_x, test_y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bbox::BBox, tests::test_image};

    #[test]
    fn test_new_test() {
        let mut dataset = MemoryDataSet::new_test();
        dataset.load();
        assert_eq!(4, dataset.samples());
    }

    #[test]
    fn test_add_sample() {
        let mut dataset = MemoryDataSet::default();
        let sample = (test_image(), vec![(BBox::default(), 0)]);
        dataset.add(sample);
        assert_eq!(0, dataset.samples());
    }

    #[test]
    fn test_load() {
        let mut dataset = MemoryDataSet::default();
        let sample = (test_image(), vec![(BBox::default(), 0)]);
        dataset.add(sample);
        assert_eq!(0, dataset.samples());
        dataset.load();
        assert_eq!(1, dataset.samples());
    }

    #[test]
    fn test_generate_annotations() {
        let mut dataset = MemoryDataSet::default();
        let sample = (test_image(), vec![(BBox::default(), 0)]);
        dataset.add(sample);
        assert_eq!(0, dataset.samples());
        dataset.load();
        assert_eq!(1, dataset.samples());
        dataset.generate_random_annotations(10);
        assert_eq!(11, dataset.samples());
    }

    #[test]
    fn test_get_data() {
        let mut dataset = MemoryDataSet::default();
        let annotation = (
            BBox {
                x: 0.0,
                y: 0.0,
                w: 5.0,
                h: 5.0,
            },
            0,
        );
        let sample = (test_image(), vec![annotation]);
        dataset.add(sample);
        dataset.load();
        let (train_x, train_y, test_x, test_y) = dataset.get();
        assert_eq!(train_x.len(), train_y.len());
        assert_eq!(train_x.len(), test_x.len());
        assert_eq!(32, train_x[0].width());
        assert_eq!(32, train_x[0].height());
        assert_eq!(annotation.1, test_y[0]);
    }
}
