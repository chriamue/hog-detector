use image::DynamicImage;
use object_detector_rust::prelude::{BriefFeature, Feature, HOGFeature};

/// defines a feature descriptor combination
#[derive(Debug)]
pub struct CombinedFeatureDescriptor {
    hog: HOGFeature,
    #[cfg(feature = "brief")]
    brief: BriefFeature,
}

impl Default for CombinedFeatureDescriptor {
    fn default() -> Self {
        CombinedFeatureDescriptor {
            hog: HOGFeature::default(),
            #[cfg(feature = "brief")]
            brief: BriefFeature::default(),
        }
    }
}

impl Feature for CombinedFeatureDescriptor {
    fn extract(&self, image: &DynamicImage) -> Result<Vec<f32>, String> {
        let data = self.hog.extract(&image)?;
        #[cfg(feature = "brief")]
        let data = [self.brief.extract(&image)?, data].concat();
        Ok(data)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use object_detector_rust::tests::test_image;

    #[test]
    fn test_default() {
        let img = test_image();
        let descriptor = CombinedFeatureDescriptor::default();
        let features = descriptor.extract(&img);

        #[cfg(feature = "brief")]
        assert_eq!(18688, features.unwrap().len());
        #[cfg(not(feature = "brief"))]
        assert_eq!(18432, features.unwrap().len());
    }
}
