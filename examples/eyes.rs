#[cfg(not(feature = "eyes"))]
fn main() {
    eprintln!("mnist example needs mnist feature: cargo run --features eyes --example eyes");
}

#[cfg(feature = "eyes")]
fn main() {
    use hog_detector::{
        classifier::SVMClassifier, dataset::EyesDataSet, DataSet, Detector, HogDetector, Trainable,
    };
    let mut model = HogDetector::<SVMClassifier>::default();

    let mut dataset = EyesDataSet::default();
    println!("downloading eyes dataset");
    dataset.load();
    println!("training eyes detector model");
    model.train_class(&dataset, 1);
    assert!(model.classifier.is_some());
    let eyes_model_file = "res/eyes_model.json";
    std::fs::write(eyes_model_file, serde_json::to_string(&model).unwrap()).unwrap();
    println!("saving model to {}", eyes_model_file);
    let model = {
        let model = std::fs::read_to_string(eyes_model_file).unwrap();
        serde_json::from_str::<HogDetector<SVMClassifier>>(&model).unwrap()
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
