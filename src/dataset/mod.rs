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

#[cfg(feature = "eyes")]
#[cfg(not(target_arch = "wasm32"))]
pub use eyes_dataset::EyesDataSet;
#[cfg(not(target_arch = "wasm32"))]
pub use folder_dataset::FolderDataSet;
pub use memory_dataset::MemoryDataSet;
#[cfg(feature = "mnist")]
#[cfg(not(target_arch = "wasm32"))]
pub use mnist_dataset::MnistDataSet;
