// source: https://github.com/tiruss/eye_detector
// https://github.com/tiruss/eye_detector/archive/refs/heads/master.zip

use crate::dataset::{rotated_frames, scaled_frames, window_crop, DataSet};
use image::io::Reader as ImageReader;
use image::{imageops::resize, imageops::FilterType, DynamicImage};
use image::{open, RgbImage};
use rand::prelude::ThreadRng;
use rand::Rng;
use std::fs::File;
use std::io::prelude::*;
use std::io::Cursor;
use std::io::{self, BufRead, SeekFrom};
use std::vec;

pub struct EyeDataSet {
    zip_url: String,
    pos_path: String,
    neg_path: String,
    pub data: Vec<(u32, RgbImage)>,
    window_size: u32,
}

impl EyeDataSet {
    pub fn new(zip_url: String, pos_path: String, neg_path: String, window_size: u32) -> Self {
        Self {
            zip_url,
            pos_path,
            neg_path,
            data: Vec::new(),
            window_size,
        }
    }

    fn download_zip(&self) -> Result<bytes::Bytes, reqwest::Error> {
        let response = reqwest::blocking::get(&self.zip_url).unwrap();
        response.bytes()
    }

    fn unzip(&mut self) {
        let downloaded = self.download_zip().unwrap();
        let buff = Cursor::new(downloaded);
        let mut archive = zip::ZipArchive::new(buff).unwrap();

        for i in 0..archive.len() {
            let mut file = archive.by_index(i).unwrap();
            let filename = file.enclosed_name().unwrap();
            if !filename.to_str().unwrap().ends_with('/') {
                if filename.to_str().unwrap().contains(&self.pos_path) {
                    let mut buff = Cursor::new(Vec::new());
                    io::copy(&mut file, &mut buff).unwrap();
                    buff.seek(SeekFrom::Start(0)).unwrap();
                    let img = ImageReader::new(buff)
                        .with_guessed_format()
                        .unwrap()
                        .decode()
                        .unwrap();
                    let img = resize(
                        &img,
                        self.window_size,
                        self.window_size,
                        FilterType::Nearest,
                    );
                    self.data.push((1, DynamicImage::ImageRgba8(img).to_rgb8()));
                } else if filename.to_str().unwrap().contains(&self.neg_path) {
                    let mut buff = Cursor::new(Vec::new());
                    io::copy(&mut file, &mut buff).unwrap();
                    buff.seek(SeekFrom::Start(0)).unwrap();
                    let img = ImageReader::new(buff)
                        .with_guessed_format()
                        .unwrap()
                        .decode()
                        .unwrap();
                    let img = resize(
                        &img,
                        self.window_size,
                        self.window_size,
                        FilterType::Nearest,
                    );
                    self.data.push((0, DynamicImage::ImageRgba8(img).to_rgb8()));
                }
            }
        }
    }

    pub fn load_annotation(
        image_path: String,
        label: String,
        x: u32,
        y: u32,
        window_size: u32,
    ) -> (String, RgbImage) {
        let img = open(image_path).unwrap().to_rgb8();
        let window = window_crop(&img, window_size, window_size, (x, y));
        (label, window)
    }

    pub fn load_annotations(
        pathes: Vec<(String, String)>,
        window_size: u32,
        augment: bool,
    ) -> Vec<(String, RgbImage)> {
        let mut annotations = Vec::new();
        for path in pathes {
            let file = File::open(path.0).unwrap();
            for line in io::BufReader::new(file).lines() {
                match line {
                    Ok(line) => {
                        let mut l = line.split(' ');
                        let label = l.next().unwrap();
                        let x: u32 = l.next().unwrap().parse().unwrap();
                        let y: u32 = l.next().unwrap().parse().unwrap();
                        match augment {
                            true => {
                                let annotation = Self::load_annotation(
                                    path.1.clone(),
                                    label.to_string(),
                                    x,
                                    y,
                                    window_size,
                                );
                                let frame = annotation.1.clone();
                                let frames = std::iter::once(&frame)
                                    .cloned()
                                    .chain(rotated_frames(&frame))
                                    .chain(scaled_frames(&frame));
                                let augmented_annotations =
                                    frames.map(|f| (annotation.0.clone(), f));
                                annotations.extend(augmented_annotations);
                            }
                            false => {
                                annotations.push(Self::load_annotation(
                                    path.1.clone(),
                                    label.to_string(),
                                    x,
                                    y,
                                    window_size,
                                ));
                            }
                        };
                    }
                    _ => (),
                }
            }
        }
        annotations
    }

    pub fn generate_random_annotations_from_image(
        image: &RgbImage,
        label: String,
        count: usize,
        window_size: u32,
    ) -> Vec<(String, RgbImage)> {
        let mut annotations = Vec::new();
        let mut rng: ThreadRng = rand::thread_rng();

        for _ in 0..count {
            let x = rng.gen_range(0..=image.width());
            let y = rng.gen_range(0..=image.height());
            annotations.push((
                label.to_string(),
                window_crop(image, window_size, window_size, (x, y)),
            ));
        }
        annotations
    }

    pub fn label_props(label: &str, labels: &[String]) -> Vec<f32> {
        let mut props = vec![0.0; 10];
        let idx = labels.iter().position(|x| x == label).unwrap();
        props[idx] = 1.0;
        props
    }

    pub fn label_id(label: &str, labels: &[String]) -> u32 {
        labels.iter().position(|x| x == label).unwrap() as u32
    }

    pub fn export(&self, folder: &str) {
        self.data
            .iter()
            .enumerate()
            .for_each(|(index, (label, img))| {
                img.save(format!("{}/{}_{}.jpg", folder, label, index))
                    .unwrap();
            });
    }
}

impl DataSet for EyeDataSet {
    fn load(&mut self, _augment: bool) {
        self.unzip();
    }

    fn generate_random_annotations(&mut self, _count_each: usize) {}

    fn get(&self) -> (Vec<RgbImage>, Vec<u32>, Vec<RgbImage>, Vec<u32>) {
        let train_x = self.data.iter().map(|(_, img)| img.clone()).collect();
        let train_y = self.data.iter().map(|(label, _)| *label).collect();

        let test_x = self.data.iter().map(|(_, img)| img.clone()).collect();
        let test_y = self.data.iter().map(|(label, _)| *label).collect();
        (train_x, train_y, test_x, test_y)
    }

    fn samples(&self) -> usize {
        self.data.len()
    }
}

impl Default for EyeDataSet {
    fn default() -> Self {
        let zip_url =
            "https://github.com/tiruss/eye_detector/archive/refs/heads/master.zip".to_string();
        let pos_data = "eye_data/eye_image/".to_string();
        let neg_data = "eye_data/noneye_image/".to_string();
        EyeDataSet::new(zip_url, pos_data, neg_data, 32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Detector;
    use crate::HogDetector;
    use crate::Trainable;

    #[test]
    fn test_default() {
        let zip_url =
            "https://github.com/tiruss/eye_detector/archive/refs/heads/master.zip".to_string();
        let dataset = EyeDataSet::default();
        assert_eq!(dataset.zip_url, zip_url);
    }

    #[ignore = "prevent download"]
    #[test]
    fn test_download() {
        let dataset = EyeDataSet::default();
        let downloaded = dataset.download_zip().unwrap();
        println!("{:?}", downloaded.len());
        assert!(downloaded.len() > 40_000_000);
    }

    #[ignore = "prevent download"]
    #[test]
    fn test_unzip() {
        let mut dataset = EyeDataSet::default();
        dataset.unzip();
        assert!(dataset.data.len() > 0);
    }

    #[ignore = "takes more than 200s in debug mode"]
    #[test]
    fn test_train() {
        let mut model = HogDetector::default();

        let mut dataset = EyeDataSet::default();
        dataset.load(false);
        model.train_class(&dataset, 1);
        assert!(model.svc.is_some());

        std::fs::write(
            "res/eyes_model.json",
            serde_json::to_string(&model).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn test_detect() {
        let model = {
            let model = std::fs::read_to_string("res/eyes_model.json").unwrap();
            serde_json::from_str::<HogDetector>(&model).unwrap()
        };
        assert!(model.svc.is_some());
        let lenna = image::open("res/lenna.png").unwrap();
        model
            .visualize_detections(&lenna)
            .save("out/test_lenna_eyes.png")
            .unwrap();
    }
}