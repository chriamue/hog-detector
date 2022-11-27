use image::DynamicImage;
use std::fmt::Debug;

mod hog_feature_descriptor;
pub use hog_feature_descriptor::HogFeatureDescriptor;

/// trait for a feature descriptor
pub trait FeatureDescriptor: Debug {
    /// calculates a feature description from a grayscale image
    fn calculate_feature(&self, image: &DynamicImage) -> Result<Vec<f32>, String>;
}
