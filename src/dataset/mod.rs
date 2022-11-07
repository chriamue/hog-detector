use image::RgbImage;

#[cfg(feature = "eyes")]
#[cfg(not(target_arch = "wasm32"))]
mod eyes_dataset;
#[cfg(not(target_arch = "wasm32"))]
mod folder_dataset;
mod memory_dataset;
#[cfg(feature = "mnist")]
#[cfg(not(target_arch = "wasm32"))]
mod mnist_dataset;

pub trait DataSet {
    fn load(&mut self, augment: bool);
    fn generate_random_annotations(&mut self, count_each: usize);
    fn samples(&self) -> usize;
    fn get(&self) -> (Vec<RgbImage>, Vec<u32>, Vec<RgbImage>, Vec<u32>);
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
