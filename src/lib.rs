#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

/// bounding box struct and functions
pub mod bbox;
/// module for classifiers
pub mod classifier;
/// data augmentation trait
pub mod data_augmentation;
/// functionality for loading train datasets
pub mod dataset;
/// detection struct module
pub mod detection;
/// detector functionality
pub mod detector;
/// feature descriptors
pub mod feature_descriptor;
/// hogdetector struct module
pub mod hogdetector;
/// predict functionality
pub mod predictable;
/// some structs and functions usable for tests
pub mod tests;
/// train functions
pub mod trainable;
/// some functions for manipulating images
pub mod utils;

/// web assembly module
//#[cfg(target_arch = "wasm32")]
#[cfg(not(tarpaulin_include))]
pub mod wasm;

pub use dataset::DataSet;
pub use detector::Detector;
pub use hogdetector::HogDetector;
pub use predictable::Predictable;
pub use trainable::Trainable;

/// object class type
pub type Class = u32;
/// annotation is a object bounding box in image and class type
pub type Annotation = (bbox::BBox, Class);

/// the hog_detector prelude
pub mod prelude {
    pub use crate::bbox::BBox;
    pub use crate::classifier::Classifier;
    pub use crate::dataset::DataSet;
    pub use crate::detection::Detection;
    pub use crate::detector::Detector;
    pub use crate::feature_descriptor::FeatureDescriptor;
    pub use crate::predictable::Predictable;
    pub use crate::trainable::Trainable;
}
