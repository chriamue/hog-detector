use image::DynamicImage;
use image::ImageOutputFormat;
use std::io::Cursor;

/// converts dynamic image to base64 encoded png image
pub fn image_to_base64_image(img: &DynamicImage) -> String {
    let mut image_data: Vec<u8> = Vec::new();
    img.write_to(&mut Cursor::new(&mut image_data), ImageOutputFormat::Png)
        .unwrap();
    let res_base64 = base64::encode(image_data);
    format!("data:image/png;base64,{}", res_base64)
}

/// decodes base64 encoded image to dynamic image
pub fn base64_image_to_image(b64img: &str) -> DynamicImage {
    let b64img = b64img.replace("data:image/png;base64,", "");
    let data = base64::decode(b64img).unwrap();
    image::load_from_memory(&data).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use imageproc::utils::rgb_bench_image;

    #[test]
    fn test_image_base64_encoding_and_decoding() {
        let image = DynamicImage::ImageRgb8(rgb_bench_image(100, 100));
        let encoded = image_to_base64_image(&image);
        assert!(encoded.starts_with("data:image/png;base64"));
        let decoded = base64_image_to_image(&encoded).to_rgb8();
        assert_eq!(image.width(), decoded.width());
        assert_eq!(image.height(), decoded.height());
    }
}
