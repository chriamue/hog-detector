use super::FeatureDescriptor;
use brief::BriefDescriptor;
use image::DynamicImage;

/// Defines a struct to represent the BRIEF descriptor
#[derive(Debug)]
pub struct BriefFeatureDescriptor {
    descriptor: BriefDescriptor,
}

impl Default for BriefFeatureDescriptor {
    fn default() -> Self {
        let descriptor = BriefDescriptor::new(256, 32);
        BriefFeatureDescriptor { descriptor }
    }
}

impl FeatureDescriptor for BriefFeatureDescriptor {
    fn calculate_feature(&self, image: &DynamicImage) -> Result<Vec<f32>, String> {
        Ok(self
            .descriptor
            .compute(&image.to_luma8())
            .iter()
            .map(|&x| x as f32)
            .collect())
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::tests::test_image;

    #[test]
    fn test_default() {
        let img = test_image();
        let descriptor = BriefFeatureDescriptor::default();
        let features = descriptor.calculate_feature(&img);
        assert_eq!(256, features.unwrap().len());
    }
}
