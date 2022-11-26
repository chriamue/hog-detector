use super::FeatureDescriptor;
use image::{ImageBuffer, Luma};
use imageproc::hog::{hog, HogOptions};

/// A Histogram of Oriented Features Descriptor
#[derive(Debug)]
pub struct HogFeatureDescriptor {
    options: HogOptions,
}

impl Default for HogFeatureDescriptor {
    fn default() -> Self {
        let options = HogOptions {
            orientations: 8,
            signed: true,
            cell_side: 4,
            block_side: 2,
            block_stride: 1,
        };
        HogFeatureDescriptor { options }
    }
}

impl FeatureDescriptor for HogFeatureDescriptor {
    fn calculate_feature(&self, image: ImageBuffer<Luma<u8>, Vec<u8>>) -> Result<Vec<f32>, String> {
        hog(&image, self.options)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::tests::test_image;
    use image::DynamicImage;

    #[test]
    fn test_default() {
        let img = DynamicImage::ImageRgb8(test_image());
        let descriptor = HogFeatureDescriptor::default();
        let features = descriptor.calculate_feature(img.to_luma8());
        assert_eq!(18432, features.unwrap().len());
    }
}
