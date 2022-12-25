use image::DynamicImage;

use super::{FeatureDescriptor, HogFeatureDescriptor};

#[cfg(feature = "brief")]
use super::BriefFeatureDescriptor;

/// defines a feature descriptor combination
#[derive(Debug)]
pub struct CombinedFeatureDescriptor {
    hog: HogFeatureDescriptor,
    #[cfg(feature = "brief")]
    brief: BriefFeatureDescriptor,
}

impl Default for CombinedFeatureDescriptor {
    fn default() -> Self {
        CombinedFeatureDescriptor {
            hog: HogFeatureDescriptor::default(),
            #[cfg(feature = "brief")]
            brief: BriefFeatureDescriptor::default(),
        }
    }
}

impl FeatureDescriptor for CombinedFeatureDescriptor {
    fn calculate_feature(&self, image: &DynamicImage) -> Result<Vec<f32>, String> {
        let data = self.hog.calculate_feature(&image)?;
        #[cfg(feature = "brief")]
        let data = [self.brief.calculate_feature(&image)?, data].concat();
        Ok(data)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::tests::test_image;

    #[test]
    fn test_default() {
        let img = test_image();
        let descriptor = CombinedFeatureDescriptor::default();
        let features = descriptor.calculate_feature(&img);

        #[cfg(feature = "brief")]
        assert_eq!(18688, features.unwrap().len());
        #[cfg(not(feature = "brief"))]
        assert_eq!(18432, features.unwrap().len());
    }
}
