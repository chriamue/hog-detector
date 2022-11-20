use image::RgbImage;
/// predictable trait
pub trait Predictable {
    /// predicts class of image
    fn predict(&self, image: &RgbImage) -> u32;
}
