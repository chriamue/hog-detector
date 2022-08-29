pub mod bbox;
pub mod dataset;
pub mod detection;
pub mod detector;
pub mod hogdetector;
pub mod predictable;
pub mod trainable;
pub mod utils;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

pub use dataset::DataSet;
pub use detector::Detector;
pub use hogdetector::HogDetector;
pub use predictable::Predictable;
pub use trainable::Trainable;

pub mod prelude {
    pub use crate::bbox::BBox;
    pub use crate::detection::Detection;
}
