use imageproc::hog::{hog, HogOptions};

pub struct Model {
    pub options: HogOptions,
}

impl Default for Model {
    fn default() -> Model {
        let options = HogOptions {
            orientations: 9,
            signed: false,
            cell_side: 8,
            block_side: 2,
            block_stride: 2,
        };
        Model { options }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{imageops::resize, imageops::FilterType, open};

    #[test]
    fn test_default() {
        let model = Model::default();
        assert_eq!(model.options.orientations, 9);
    }

    #[test]
    fn test_hog() {
        let model = Model::default();
        let loco03 = open("res/loco03.jpg").unwrap().to_luma8();
        let loco03 = resize(&loco03, 32, 32, FilterType::Nearest);
        let descriptor = hog(&loco03, model.options).unwrap();
        assert_eq!(descriptor.len(), 144);
    }
}
