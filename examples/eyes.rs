#[cfg(not(feature = "eyes"))]
fn main() {
    eprintln!("mnist example needs mnist feature: cargo run --features eyes --example eyes");
}

#[cfg(feature = "eyes")]
fn main() {
    use hog_detector::dataset::EyesDataSet;
    use hog_detector::{DataSet, HogDetector};
    use object_detector_rust::prelude::BayesClassifier;
    let mut model: HogDetector<f32, usize, BayesClassifier<f32, usize>, _> = HogDetector::default();

    let mut dataset = EyesDataSet::default();
    println!("downloading eyes dataset");
    dataset.load().unwrap();
    println!("training eyes detector model");
    let (x, y) = dataset.get_data();
    let y = y.into_iter().map(|y| y as usize).collect();
    model.fit_class(&x, &y, 1).unwrap();
    assert!(model.classifier.is_some());
    let image_file = "res/lenna.png";
    let lenna = image::open(image_file).unwrap();
    let result_file = "out/test_lenna_eyes.png";
    model
        .visualize_detections(&lenna)
        .save(result_file)
        .unwrap();
    println!("detections saved to {}", result_file);
}
