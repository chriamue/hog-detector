# hog-detector

[![Github Repo](https://img.shields.io/badge/github-repo-green)](https://github.com/chriamue/hog-detector/)
[![Github Pages Build](https://github.com/chriamue/hog-detector/actions/workflows/gh-pages.yml/badge.svg)](https://chriamue.github.io/hog-detector/)
[![Benchmarks](https://github.com/chriamue/hog-detector/actions/workflows/bench.yml/badge.svg)](https://github.com/chriamue/hog-detector/actions/workflows/bench.yml)
[![codecov](https://codecov.io/gh/chriamue/hog-detector/branch/main/graph/badge.svg?token=RJ6T5D9DZT)](https://codecov.io/gh/chriamue/hog-detector)
[![Demo](https://img.shields.io/badge/Demo-online-green.svg)](https://chriamue.github.io/hog-detector/)
[![Doc](https://img.shields.io/badge/Docs-online-green.svg)](https://chriamue.github.io/hog-detector/hog_detector/)

Histogram of Oriented Gradients and Object Detection

This project uses support vector machines (SVM) and histogram of oriented gradients (HOG) to detect objects in images. The SVM classifier is trained on HOG features extracted from training images to identify the objects in new images. The project is written in Rust and can be compiled to WebAssembly (WASM) for use in web applications.

You can find a [demo here](https://chriamue.github.io/hog-detector)

## How it works

1. A training dataset with positive and negative samples is given.
2. HOG descriptors of the samples are calculated.
3. A Linear Support Vector Machine is trained on the positive and negative samples.
4. Using a sliding window, the svm classifier detects bounding boxes.
5. Applying non-maximum suppression removes some of the bounding boxes.

## Requirements

* Rust
* wasm-pack (for compiling to WASM)

## Usage

1. Clone the repository:

```sh
git clone https://github.com/chriamue/hog-detector
cd hog-detector
```

2. Compile the code to WASM:

```sh
wasm-pack build --target web
```

3. Run the Web version in your browser

```sh
python3 -m http.server
```

Open your browser on [Localhost](http://localhost:8000)

## Examples

You can find examples in example folder.
The mnist example loads the mnist dataset and trains hog on the numbers.

```sh
cargo run --features mnist --example mnist
```

## Testing

The project includes a test suite that can be run with:

```sh
cargo test
```

Benchmarks can be run with:

```sh
cargo bench
```

## Train data preparation

Find a minimal annotation tool in the [demo](https://chriamue.github.io/hog-detector).

## References

[https://pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/](https://pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/)

[Hard Negative Mining](https://openaccess.thecvf.com/content_ECCV_2018/papers/SouYoung_Jin_Unsupervised_Hard-Negative_Mining_ECCV_2018_paper.pdf)

[Eye Dataset](https://github.com/tiruss/eye_detector/tree/master/eye_data)
