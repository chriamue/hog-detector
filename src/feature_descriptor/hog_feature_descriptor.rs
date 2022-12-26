use super::FeatureDescriptor;
use image::DynamicImage;
use imageproc::hog::{hog, HogOptions};

/// This struct is a HogFeatureDescriptor which is used to represent a Histogram of Oriented Features Descriptor.
/// It contains one field, options, which is of type HogOptions.
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
    fn calculate_feature(&self, image: &DynamicImage) -> Result<Vec<f32>, String> {
        hog(&image.to_luma8(), self.options)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::tests::test_image;

    #[test]
    fn test_default() {
        let img = test_image();
        let descriptor = HogFeatureDescriptor::default();
        let features = descriptor.calculate_feature(&img);
        assert_eq!(18432, features.unwrap().len());
    }
}
