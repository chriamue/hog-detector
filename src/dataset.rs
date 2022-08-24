use image::{imageops, open, Rgb, RgbImage};
use imageproc::geometric_transformations::{rotate_about_center, warp, Interpolation, Projection};
use rand::prelude::ThreadRng;
use rand::Rng;
use std::fs::read_dir;
use std::fs::File;
use std::io::{self, BufRead};
use std::vec;

pub fn window_crop(
    input_frame: &RgbImage,
    window_width: u32,
    window_height: u32,
    center: (u32, u32),
) -> RgbImage {
    imageops::crop(
        &mut input_frame.clone(),
        center
            .0
            .saturating_sub(window_width / 2)
            .min(input_frame.width() - window_width),
        center
            .1
            .saturating_sub(window_height / 2)
            .min(input_frame.height() - window_height),
        window_width,
        window_height,
    )
    .to_image()
}

pub fn rotated_frames(frame: &RgbImage) -> impl Iterator<Item = RgbImage> + '_ {
    [
        0.02, -0.02, 0.05, -0.05, 0.07, -0.07, 0.09, -0.09, 1.1, -1.1, 1.3, -1.3, 1.5, -1.5, 2.0,
        -2.0,
    ]
    .iter()
    .map(|rad| rotate_about_center(frame, *rad, Interpolation::Nearest, Rgb([0, 0, 0])))
}

pub fn scaled_frames(frame: &RgbImage) -> impl Iterator<Item = RgbImage> + '_ {
    [0.8, 0.9, 1.1, 1.2].into_iter().map(|scalefactor| {
        let scale = Projection::scale(scalefactor, scalefactor);

        warp(frame, &scale, Interpolation::Nearest, Rgb([0, 0, 0]))
    })
}

pub struct DataSet {
    path: String,
    pub data: Vec<(String, RgbImage)>,
    names: Vec<String>,
    window_size: u32,
}

impl DataSet {
    pub fn new(path: String, label_names_path: String, window_size: u32) -> DataSet {
        DataSet {
            path,
            data: Vec::new(),
            names: Self::load_label_names(label_names_path),
            window_size,
        }
    }

    pub fn load_label_names(path: String) -> Vec<String> {
        let file = File::open(path).unwrap();
        io::BufReader::new(file)
            .lines()
            .map(|line| line.unwrap())
            .collect()
    }

    pub fn load(&mut self, augment: bool) {
        let pathes = Self::list_pathes(&self.path);
        let annotations = Self::load_annotations(pathes, self.window_size, augment);
        self.data = annotations;
    }

    pub fn list_pathes(path: &str) -> Vec<(String, String)> {
        let mut file_pathes = Vec::new();
        for entry in read_dir(path).unwrap() {
            let path = entry.unwrap();
            if path.path().to_str().unwrap().ends_with(".jpg") {
                let image_path = path.path();
                let image_path = image_path.as_path().to_str().unwrap();
                let labels_path = image_path.replace("jpg", "txt");
                file_pathes.push((labels_path.to_string(), image_path.to_string()));
            }
        }
        file_pathes
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

    pub fn generate_random_annotations(&mut self, count_each: usize) {
        let pathes = Self::list_pathes(&self.path);
        for (_, image_path) in pathes {
            let img = open(image_path).unwrap().to_rgb8();
            self.data
                .extend(Self::generate_random_annotations_from_image(
                    &img,
                    "none".to_string(),
                    count_each,
                    self.window_size,
                ));
        }
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

    pub fn get(&self) -> (Vec<RgbImage>, Vec<u32>, Vec<RgbImage>, Vec<u32>) {
        let train_x = self.data.iter().map(|(_, img)| img.clone()).collect();
        let train_y = self
            .data
            .iter()
            .map(|(label, _)| Self::label_id(label, &self.names))
            .collect();

        let test_x = self.data.iter().map(|(_, img)| img.clone()).collect();
        let test_y = self
            .data
            .iter()
            .map(|(label, _)| Self::label_id(label, &self.names))
            .collect();
        (train_x, train_y, test_x, test_y)
    }

    pub fn samples(&self) -> usize {
        self.data.len()
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

#[cfg(test)]
mod tests {
    use super::*;

    const LABELS: usize = 18;
    const IMAGES_PER_LABEL: usize = 21;

    #[test]
    fn test_list_files() {
        let path = "res/training";
        let file_pathes = DataSet::list_pathes(path);
        assert_eq!(file_pathes.len(), 3);
    }

    #[test]
    fn test_load_annotations() {
        let pathes = vec![(
            "res/training/webcam01.txt".to_string(),
            "res/training/webcam01.jpg".to_string(),
        )];
        let annotations = DataSet::load_annotations(pathes, 28, false);
        assert_eq!(annotations.len(), 6);
    }

    #[test]
    fn test_dataset() {
        let mut dataset = DataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            28,
        );
        dataset.load(false);
        assert_eq!(dataset.samples(), 18);
    }

    #[test]
    fn test_load_label_names() {
        let labels = DataSet::load_label_names("res/labels.txt".to_string());
        assert_eq!(labels.len(), 10);
        assert_eq!(labels[5], "loco5");
        assert_eq!(labels.into_iter().position(|x| x == "loco5"), Some(5));
    }

    #[test]
    fn test_label_props() {
        let labels = DataSet::load_label_names("res/labels.txt".to_string());
        let props = DataSet::label_props("loco5", &labels);
        assert_eq!(props.len(), 10);
        assert_eq!(props[5], 1.0);
        assert_eq!(props[0], 0.0);
    }

    #[test]
    fn test_load_augmented() {
        let mut dataset = DataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            28,
        );
        dataset.load(true);
        assert_eq!(dataset.samples(), LABELS * IMAGES_PER_LABEL);
    }

    #[test]
    fn test_generate_random_annotations() {
        let image = RgbImage::new(32, 32);
        let annotations =
            DataSet::generate_random_annotations_from_image(&image, "none".to_string(), 5, 28);
        assert_eq!(annotations.len(), 5);
        assert_eq!(annotations.last().unwrap().0, "none");

        let mut dataset = DataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            28,
        );
        dataset.load(true);
        assert_eq!(dataset.samples(), LABELS * IMAGES_PER_LABEL);
        dataset.generate_random_annotations(1);
        assert_eq!(dataset.samples(), LABELS * IMAGES_PER_LABEL + 3);
    }

    #[test]
    fn test_dataset_export() {
        let mut dataset = DataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            28,
        );
        dataset.load(true);
        dataset.export("out/export");
    }
}
