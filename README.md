<div id="top"></div>
  <h3 align="center">Audio Adversarial Example</h3>

## Usage

1. train.py: train and test models
2. exp_ags.py: generate and test AGS steganography method
3. exp_adversarial.py: generate adversarial example to elevate detection rate for cover
4. exp_face_gradient_img.py: generate gradient image on image domain

## Supported Files

7. filters.py: rich model
8. gradient.py: get gradient
9. steganography.py: multiple modify methods based on data and gradient
10. utils.py: miscellaneous small tools
11. network.py: RHFCN and WASDN model
12. config.py: config dataset address
13. dataset.py: generate cover-stego pairing dataset
14. ut_*.py: unit test files