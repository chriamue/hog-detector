# hog-detector

[![Github Repo](https://img.shields.io/badge/github-repo-green)](https://github.com/chriamue/hog-detector/)
[![Github Pages Build](https://github.com/chriamue/hog-detector/actions/workflows/gh-pages.yml/badge.svg)](https://chriamue.github.io/hog-detector/)
[![Benchmarks](https://github.com/chriamue/hog-detector/actions/workflows/bench.yml/badge.svg)](https://github.com/chriamue/hog-detector/actions/workflows/bench.yml)
[![Demo](https://img.shields.io/badge/Demo-online-green.svg)](https://chriamue.github.io/hog-detector/)
[![Doc](https://img.shields.io/badge/Docs-online-green.svg)](https://chriamue.github.io/hog-detector/hog_detector/)

Histogram of Oriented Gradients and Object Detection

This project contains an algorithm for object detection based on SVM on HOG descriptors.

You can find a [demo here](https://chriamue.github.io/hog-detector)

## how it works

1. A training dataset with positive and negative samples is given.
2. HOG descriptors of the samples are calculated.
3. A Linear Support Vector Machine is trained on the positive and negative samples.
4. Using a sliding window, the svm classifier detects bounding boxes.
5. Applying non-maximum suppression removes some of the bounding boxes.

## run examples

You can find examples in example folder.
The mnist example loads the mnist dataset and trains hog on the numbers.

```sh
cargo run --features mnist --example mnist
```

## run wasm

```sh
wasm-pack build --target web
python3 -m http.server
```

## run tests

You can run tests

```sh
cargo test
```

and you can run benchmarks

```sh
cargo bench
```

## train data preparation

Find a minimal annotation tool in the [demo](https://chriamue.github.io/hog-detector).

## references

[https://pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/](https://pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/)

[Hard Negative Mining](https://openaccess.thecvf.com/content_ECCV_2018/papers/SouYoung_Jin_Unsupervised_Hard-Negative_Mining_ECCV_2018_paper.pdf)

[Eye Dataset](https://github.com/tiruss/eye_detector/tree/master/eye_data)
