use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::RgbImage;

fn rotated_frames(image_size: u32) {
    let image = RgbImage::new(image_size, image_size);
    let _frames: Vec<RgbImage> = hog_detector::utils::rotated_frames(&image).collect();
}

fn scaled_frames(image_size: u32) {
    let image = RgbImage::new(image_size, image_size);
    let _frames: Vec<RgbImage> = hog_detector::utils::scaled_frames(&image).collect();
}

fn window_crop(window_size: u32) {
    let image = RgbImage::new(100, 100);
    let _cropped = hog_detector::utils::window_crop(&image, window_size, window_size, (20, 20));
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("rotated_frames 20", |b| {
        b.iter(|| rotated_frames(black_box(32)))
    });
    c.bench_function("scaled_frames 20", |b| {
        b.iter(|| scaled_frames(black_box(32)))
    });
    c.bench_function("window_crop 20", |b| b.iter(|| window_crop(black_box(32))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
