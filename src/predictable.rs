use image::DynamicImage;
/// predictable trait
pub trait Predictable {
    /// predicts class of image
    fn predict(&self, image: &DynamicImage) -> u32;
}
