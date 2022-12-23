use image::DynamicImage;
use std::fmt::Debug;

#[cfg(feature = "brief")]
mod brief_feature_descriptor;
mod combined_feature_descriptor;
mod hog_feature_descriptor;

#[cfg(feature = "brief")]
pub use brief_feature_descriptor::BriefFeatureDescriptor;
pub use combined_feature_descriptor::CombinedFeatureDescriptor;
pub use hog_feature_descriptor::HogFeatureDescriptor;

/// trait for a feature descriptor
pub trait FeatureDescriptor: Debug {
    /// calculates a feature description from a grayscale image
    fn calculate_feature(&self, image: &DynamicImage) -> Result<Vec<f32>, String>;
}
