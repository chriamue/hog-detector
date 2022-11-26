#[cfg(not(feature = "eyes"))]
fn main() {
    eprintln!("mnist example needs mnist feature: cargo run --features eyes --example eyes");
}

#[cfg(feature = "eyes")]
fn main() {
    use hog_detector::{
        classifier::SVMClassifier, dataset::EyesDataSet, hogdetector::HogDetectorTrait, DataSet,
        Detector, HogDetector, Trainable,
    };
    let mut model = HogDetector::<SVMClassifier>::default();

    let mut dataset = EyesDataSet::default();
    println!("downloading eyes dataset");
    dataset.load();
    println!("training eyes detector model");
    model.train_class(&dataset, 1);
    assert!(model.classifier.is_some());
    let eyes_model_file = "res/eyes_model.json";
    std::fs::write(eyes_model_file, model.save()).unwrap();
    println!("saving model to {}", eyes_model_file);
    let model = {
        let mut model = HogDetector::<SVMClassifier>::default();
        model.load(&std::fs::read_to_string(eyes_model_file).unwrap());
        model
    };
    assert!(model.classifier.is_some());
    let image_file = "res/lenna.png";
    println!("detecting eyes on image {}", eyes_model_file);
    let lenna = image::open(image_file).unwrap();
    let result_file = "out/test_lenna_eyes.png";
    model
        .visualize_detections(&lenna)
        .save(result_file)
        .unwrap();
    println!("detections saved to {}", result_file);
}
