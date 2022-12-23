use image::DynamicImage;

use super::{FeatureDescriptor, HogFeatureDescriptor};

#[cfg(feature = "brief")]
use super::BriefFeatureDescriptor;

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
