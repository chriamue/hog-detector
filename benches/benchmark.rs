use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use hog_detector::{
    classifier::BayesClassifier,
    feature_descriptor::{FeatureDescriptor, HogFeatureDescriptor},
    tests::test_image,
    HogDetector,
};
use image::{DynamicImage, RgbImage};

fn rotated_frames(b: &mut Bencher) {
    let image = DynamicImage::ImageRgb8(RgbImage::new(32, 32));
    b.iter(|| {
        let _frames: Vec<DynamicImage> = hog_detector::utils::rotated_frames(&image).collect();
    })
}

fn scaled_frames(b: &mut Bencher) {
    let image = DynamicImage::ImageRgb8(RgbImage::new(32, 32));
    b.iter(|| {
        let _frames: Vec<DynamicImage> = hog_detector::utils::scaled_frames(&image).collect();
    })
}

fn window_crop(b: &mut Bencher) {
    let image = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
    b.iter(|| {
        let _cropped = hog_detector::utils::window_crop(&image, 32, 32, (20, 20));
    })
}

fn calculate_hog_feature(b: &mut Bencher) {
    let image = test_image();
    let feature_descriptor = HogFeatureDescriptor::default();
    b.iter(|| {
        feature_descriptor.calculate_feature(&image).unwrap();
    })
}

fn preprocess_matrix(b: &mut Bencher) {
    let image = test_image();
    let images = vec![image.clone(), image.clone(), image.clone(), image.clone()];
    let hog_detector: HogDetector<BayesClassifier> = HogDetector::default();
    b.iter(|| hog_detector.preprocess_matrix(images.clone()));
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("rotated_frames", rotated_frames);
    c.bench_function("scaled_frames", scaled_frames);
    c.bench_function("window_crop", window_crop);
    c.bench_function("calculate hog feature", calculate_hog_feature);
    c.bench_function("preprocess matrix", preprocess_matrix);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
