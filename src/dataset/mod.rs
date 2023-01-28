pub use object_detector_rust::dataset::{AnnotatedImageSet, DataSet};

#[cfg(feature = "eyes")]
#[cfg(not(target_arch = "wasm32"))]
mod eyes_dataset;
#[cfg(feature = "mnist")]
#[cfg(not(target_arch = "wasm32"))]
mod mnist_dataset;

#[cfg(feature = "eyes")]
#[cfg(not(target_arch = "wasm32"))]
pub use eyes_dataset::EyesDataSet;
#[cfg(feature = "mnist")]
#[cfg(not(target_arch = "wasm32"))]
pub use mnist_dataset::MnistDataSet;
