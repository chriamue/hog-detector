use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use object_detector_rust::prelude::{BBox, Feature, HOGFeature};
use object_detector_rust::tests::test_image;
use object_detector_rust::utils::crop_bbox;

fn calculate_hog_feature(b: &mut Bencher) {
    let image = test_image();
    let image = crop_bbox(&image, &BBox::new(0, 0, 64, 64));
    let feature_descriptor = HOGFeature::default();
    b.iter(|| {
        feature_descriptor.extract(&image).unwrap();
    })
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("calculate hog feature", calculate_hog_feature);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
