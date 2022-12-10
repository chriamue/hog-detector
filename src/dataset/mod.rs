use crate::Annotation;
use crate::Detector;
use image::DynamicImage;

#[cfg(feature = "eyes")]
#[cfg(not(target_arch = "wasm32"))]
mod eyes_dataset;
#[cfg(not(target_arch = "wasm32"))]
mod folder_dataset;
mod memory_dataset;
#[cfg(feature = "mnist")]
#[cfg(not(target_arch = "wasm32"))]
mod mnist_dataset;

/// Image annotated by list of Annotations
pub type AnnotatedImage = (DynamicImage, Vec<Annotation>);

/// trait for a dataset
pub trait DataSet {
    /// loads the dataset
    fn load(&mut self);
    /// generates random annotations
    fn generate_random_annotations(&mut self, count_each: usize);
    /// number of samples in dataset
    fn samples(&self) -> usize;
    /// get train and test data
    fn get(&self) -> (Vec<DynamicImage>, Vec<u32>, Vec<DynamicImage>, Vec<u32>);
}

/// trait of a set of annotated images
pub trait AnnotatedImageSet {
    /// adds an annotated image
    fn add_annotated_image(&mut self, annotated_image: AnnotatedImage);
    /// returns count of annotated images
    fn annotated_images_size(&self) -> usize;
    /// returns iterator over annotated images
    fn annotated_images(&self) -> Box<dyn Iterator<Item = &AnnotatedImage> + '_>;
}

/// trait for generating data
pub trait DataGenerator: AnnotatedImageSet {
    /// generates hard negative samples, see: [Hard Negative Mining](https://openaccess.thecvf.com/content_ECCV_2018/papers/SouYoung_Jin_Unsupervised_Hard-Negative_Mining_ECCV_2018_paper.pdf)
    fn generate_hard_negative_samples(
        &mut self,
        detector: &dyn Detector,
        class: u32,
        max_annotations: Option<usize>,
    ) {
        let annotated_images = self.generate_negative_samples(detector, class, max_annotations);
        annotated_images
            .into_iter()
            .for_each(|annotated_image| self.add_annotated_image(annotated_image));
    }

    /// generates negative samples
    fn generate_negative_samples(
        &self,
        detector: &dyn Detector,
        class: u32,
        max_annotations: Option<usize>,
    ) -> Vec<AnnotatedImage> {
        let mut annotations_counter = 0;
        let mut generated_annotated_images = Vec::new();
        for annotated_image in self.annotated_images() {
            let detections = detector.detect_objects(&annotated_image.0);
            let mut false_pos_annotations = Vec::new();
            detections.iter().for_each(|detection| {
                if max_annotations.is_some() && max_annotations.unwrap() <= annotations_counter {
                    return;
                }
                if detection.class as u32 == class {
                    let mut false_pos = true;
                    annotated_image
                        .1
                        .iter()
                        .for_each(|(bbox, annotated_class)| {
                            if class == *annotated_class {
                                if bbox.iou(&detection.bbox) > 0.1 {
                                    false_pos = false;
                                }
                            }
                        });
                    if false_pos {
                        let false_pos_bbox = detection.bbox.clone();
                        let false_pos_annotation: Annotation = (false_pos_bbox, 0);
                        false_pos_annotations.push(false_pos_annotation);
                        annotations_counter += 1;
                    }
                }
            });
            generated_annotated_images.push((annotated_image.0.clone(), false_pos_annotations));
        }
        generated_annotated_images
    }
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
    use crate::bbox::BBox;

    #[test]
    fn test_hard_negative_samples() {
        use crate::prelude::Detection;
        use mockall::*;

        mock! {
            HogDetector {}
            impl Detector for HogDetector {
                fn detect_objects(&self, image: &image::DynamicImage) -> Vec<crate::prelude::Detection>;
                fn visualize_detections(&self, image: &image::DynamicImage) -> image::DynamicImage;
            }
        }

        let mut model = MockHogDetector::new();
        model.expect_detect_objects().returning(move |_| {
            vec![Detection {
                bbox: BBox {
                    x: 0.0,
                    y: 0.0,
                    w: 50.0,
                    h: 50.0,
                },
                class: 1,
                confidence: 1.0,
            }]
        });

        let mut dataset = MemoryDataSet::new_test();
        dataset.load();
        let samples = dataset.samples();
        let annotated_images_size = dataset.annotated_images_size();
        dataset.generate_hard_negative_samples(&model, 1, Some(1));
        dataset.load();
        assert!(dataset.samples() > samples);
        assert!(dataset.annotated_images_size() > annotated_images_size);
    }
}
