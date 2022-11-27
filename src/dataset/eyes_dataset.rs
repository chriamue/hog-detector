// source: https://github.com/tiruss/eye_detector
// https://github.com/tiruss/eye_detector/archive/refs/heads/master.zip

use crate::dataset::DataSet;
use crate::utils::window_crop;
use image::io::Reader as ImageReader;
use image::{imageops::resize, imageops::FilterType, DynamicImage};
use rand::prelude::ThreadRng;
use rand::Rng;
use std::io::prelude::*;
use std::io::Cursor;
use std::io::{self, SeekFrom};
use std::vec;

/// dataset of eyes images from [tiruss/eye_detector](https://github.com/tiruss/eye_detector/)
pub struct EyesDataSet {
    zip_url: String,
    pos_path: String,
    neg_path: String,
    data: Vec<(u32, DynamicImage)>,
    window_size: u32,
}

impl EyesDataSet {
    /// construct new EyesDataSet
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
                    self.data.push((1, DynamicImage::ImageRgba8(img)));
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
                    self.data.push((0, DynamicImage::ImageRgba8(img)));
                }
            }
        }
    }

    /// generates random annotations from an image
    pub fn generate_random_annotations_from_image(
        image: &DynamicImage,
        label: String,
        count: usize,
        window_size: u32,
    ) -> Vec<(String, DynamicImage)> {
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

    /// get vector of labels with given label as 1
    pub fn label_props(label: &str, labels: &[String]) -> Vec<f32> {
        let mut props = vec![0.0; 10];
        let idx = labels.iter().position(|x| x == label).unwrap();
        props[idx] = 1.0;
        props
    }

    /// get id of label from labels
    pub fn label_id(label: &str, labels: &[String]) -> u32 {
        labels.iter().position(|x| x == label).unwrap() as u32
    }

    /// export dataset to given folder
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

impl DataSet for EyesDataSet {
    fn load(&mut self) {
        self.unzip();
    }

    fn generate_random_annotations(&mut self, _count_each: usize) {}

    fn get(&self) -> (Vec<DynamicImage>, Vec<u32>, Vec<DynamicImage>, Vec<u32>) {
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

impl Default for EyesDataSet {
    fn default() -> Self {
        let zip_url =
            "https://github.com/tiruss/eye_detector/archive/refs/heads/master.zip".to_string();
        let pos_data = "eye_data/eye_image/".to_string();
        let neg_data = "eye_data/noneye_image/".to_string();
        EyesDataSet::new(zip_url, pos_data, neg_data, 32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classifier::{BayesClassifier, CombinedClassifier, RandomForestClassifier};
    use crate::hogdetector::HogDetectorTrait;
    use crate::Detector;
    use crate::HogDetector;
    use crate::Trainable;

    #[test]
    fn test_default() {
        let zip_url =
            "https://github.com/tiruss/eye_detector/archive/refs/heads/master.zip".to_string();
        let dataset = EyesDataSet::default();
        assert_eq!(dataset.zip_url, zip_url);
    }

    #[ignore = "prevent download"]
    #[test]
    fn test_download() {
        let dataset = EyesDataSet::default();
        let downloaded = dataset.download_zip().unwrap();
        println!("{:?}", downloaded.len());
        assert!(downloaded.len() > 40_000_000);
    }

    #[ignore = "prevent download"]
    #[test]
    fn test_unzip() {
        let mut dataset = EyesDataSet::default();
        dataset.unzip();
        assert!(dataset.data.len() > 0);
    }

    #[ignore = "takes more than 200s in debug mode"]
    #[test]
    fn test_train_svm_model() {
        let mut model = HogDetector::svm();

        let mut dataset = EyesDataSet::default();
        dataset.load();
        model.train_class(&dataset, 1);
        assert!(model.classifier.is_some());

        std::fs::write("res/eyes_svm_model.json", model.save()).unwrap();
    }

    #[ignore = "takes more than 200s in debug mode"]
    #[test]
    fn test_train_random_forest_model() {
        let mut model = HogDetector::<RandomForestClassifier>::default();

        let mut dataset = EyesDataSet::default();
        dataset.load();
        model.train_class(&dataset, 1);
        assert!(model.classifier.is_some());

        std::fs::write("res/eyes_random_forest_model.json", model.save()).unwrap();
    }

    #[ignore = "takes more than 200s in debug mode"]
    #[test]
    fn test_train_bayes_model() {
        let mut model = HogDetector::<BayesClassifier>::default();

        let mut dataset = EyesDataSet::default();
        dataset.load();
        model.train_class(&dataset, 1);
        assert!(model.classifier.is_some());

        std::fs::write("res/eyes_bayes_model.json", model.save()).unwrap();
    }

    //#[ignore = "takes more than 200s in debug mode"]
    #[test]
    fn test_train_combined_model() {
        let mut model = HogDetector::<CombinedClassifier>::default();

        let mut dataset = EyesDataSet::default();
        dataset.load();
        model.train_class(&dataset, 1);
        assert!(model.classifier.is_some());

        std::fs::write("res/eyes_combined_model.json", model.save()).unwrap();
    }

    #[test]
    fn test_detect() {
        let model = {
            let mut model = HogDetector::<RandomForestClassifier>::default();
            model.load(&std::fs::read_to_string("res/eyes_random_forest_model.json").unwrap());
            model
        };
        assert!(model.classifier.is_some());
        let lenna = image::open("res/lenna.png").unwrap();
        model
            .visualize_detections(&lenna)
            .save("out/test_lenna_eyes.png")
            .unwrap();
    }
}
