use crate::{bbox::BBox, DataSet};
use image_label_tool::prelude::LabelTool;

use crate::dataset::{AnnotatedImageSet, MemoryDataSet};

/// creates a memory database from images in label tool
pub fn create_dataset(label_tool: &LabelTool) -> MemoryDataSet {
    // get all images from label tool
    let images = label_tool.annotated_images().lock().unwrap().clone();
    // create memory dataset
    let mut dataset = MemoryDataSet::default();
    for image in images.iter() {
        let img = image.get_image();
        let annotations: Vec<(BBox, u32)> = image
            .get_annotations()
            .iter()
            .map(|(bbox, class)| {
                (
                    BBox {
                        x: bbox.x,
                        y: bbox.y,
                        w: bbox.w,
                        h: bbox.h,
                    },
                    *class,
                )
            })
            .collect();

        dataset.add_annotated_image((img, annotations));
    }
    dataset.load();
    dataset.generate_random_annotations(10);
    dataset
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, ImageBuffer};
    use image_label_tool::prelude::AnnotatedImage;

    use crate::DataSet;
    use image_label_tool::prelude::BBox;

    use super::*;

    #[test]
    fn test_create_dataset() {
        let annotations = AnnotatedImage::new();
        annotations.set_image(DynamicImage::ImageRgba8(ImageBuffer::new(100, 100)));
        annotations.push((
            BBox {
                x: 0.0,
                y: 0.0,
                w: 10.0,
                h: 10.0,
            },
            1,
        ));
        let label_tool = LabelTool::new();
        label_tool.push(annotations);
        let dataset = create_dataset(&label_tool);
        assert_eq!(dataset.samples(), 11);
    }
}
