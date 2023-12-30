use image_label_tool::prelude::LabelTool;

use object_detector_rust::{
    dataset::{AnnotatedImageSet, DataSet, MemoryDataSet},
    prelude::BBox,
};

/// creates a memory database from images in label tool
pub fn create_dataset(label_tool: &LabelTool) -> MemoryDataSet {
    // get all images from label tool
    let images = label_tool.annotated_images().lock().unwrap().clone().images;
    // create memory dataset
    let mut dataset = MemoryDataSet::default();
    for image in images.borrow().iter() {
        let img = image.get_image().clone();
        let annotations: Vec<(BBox, u32)> = image
            .get_annotations()
            .iter()
            .map(|(bbox, class)| {
                (
                    BBox {
                        x: bbox.x as i32,
                        y: bbox.y as i32,
                        width: bbox.w as u32,
                        height: bbox.h as u32,
                    },
                    *class,
                )
            })
            .collect();

        dataset.add_annotated_image((img, annotations).into());
    }
    dataset.load().unwrap();
    dataset
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, ImageBuffer};
    use image_label_tool::prelude::AnnotatedImage;
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
        assert_eq!(dataset.len(), 1);
    }
}
