#![allow(warnings)]

use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct Image {
    pub rows: usize,
    pub cols: usize,
    pub R: Vec<u8>,
    pub G: Vec<u8>,
    pub B: Vec<u8>,
}

pub fn load_image(path: impl AsRef<Path>+ std::fmt::Debug) -> Image {
    let result: Image = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert!(result.rows * result.cols == result.R.len());
    result
}

pub fn store_image(image: Image, path: impl AsRef<Path>) {
    assert!(image.rows * image.cols == image.R.len());
    let _ = std::fs::write(path, serde_json::to_string_pretty(&image).unwrap());
}

pub fn main(){
}