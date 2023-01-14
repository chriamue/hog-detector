pub use object_detector_rust::dataset::{AnnotatedImageSet, DataSet};
use object_detector_rust::prelude::Detector;
use object_detector_rust::types::AnnotatedImage;
use object_detector_rust::utils::add_hard_negative_samples;
use object_detector_rust::utils::generate_negative_samples;

#[cfg(feature = "eyes")]
#[cfg(not(target_arch = "wasm32"))]
mod eyes_dataset;
#[cfg(feature = "mnist")]
#[cfg(not(target_arch = "wasm32"))]
mod mnist_dataset;

/// trait for generating data
pub trait DataGenerator: AnnotatedImageSet {
    /// generates hard negative samples, see: [Hard Negative Mining](https://openaccess.thecvf.com/content_ECCV_2018/papers/SouYoung_Jin_Unsupervised_Hard-Negative_Mining_ECCV_2018_paper.pdf)
    fn generate_hard_negative_samples(
        &mut self,
        detector: &dyn Detector,
        class: u32,
        max_annotations: Option<usize>,
    ) where
        Self: Sized,
    {
        add_hard_negative_samples(self, detector, class, max_annotations, 32, 32);
    }

    /// generates negative samples
    fn generate_negative_samples(
        &self,
        detector: &dyn Detector,
        class: u32,
        max_annotations: Option<usize>,
    ) -> Vec<AnnotatedImage>
    where
        Self: Sized,
    {
        generate_negative_samples(self, detector, class, max_annotations, 32, 32)
    }
}

#[cfg(feature = "eyes")]
#[cfg(not(target_arch = "wasm32"))]
pub use eyes_dataset::EyesDataSet;
#[cfg(feature = "mnist")]
#[cfg(not(target_arch = "wasm32"))]
pub use mnist_dataset::MnistDataSet;

#[cfg(test)]
mod tests {
    use ndarray::{Array1, ArrayView2};
    use object_detector_rust::prelude::{BBox, MemoryDataSet};

    use super::*;
    #[test]
    fn test_hard_negative_samples() {
        use crate::prelude::Detection;
        use mockall::*;
        use object_detector_rust::prelude::Predictable;

        mock! {
                HogDetector {}
                impl<X, Y> Predictable<X, Y> for HogDetectorwhere
                X: Float,
                Y: Label,
        {
            fn predict(&self, x: &ArrayView2<X>) -> Result<Array1<Y>, String> {
                Ok(vec![0])
            }

            }
        }

        let mut model = MockHogDetector::new();
        model.expect_detect_objects().returning(move |_| {
            vec![Detection {
                bbox: BBox {
                    x: 0.0,
                    y: 0.0,
                    width: 50,
                    height: 50,
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
