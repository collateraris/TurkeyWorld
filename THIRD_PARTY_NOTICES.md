# Third-party notices

The compact VQGAN follows the encoder/codebook/decoder and adversarial perceptual
training design of CompVis Taming Transformers. It is adapted for 128px game
frames and is not compatible with official pretrained codec checkpoints.

- Project: https://compvis.github.io/taming-transformers/
- Reference: https://github.com/CompVis/taming-transformers
- Inspected revision: `3ba01b241669f5ade541ce990f7650a3b8f65318`

The reference implementation is distributed under the following MIT license:

```text
Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE./

```

The optional LPIPS/VGG loss uses separately distributed pretrained loss-network
weights. They are frozen, used only during codec training, and excluded from
our codec and world-model inference exports. Codec and PatchGAN weights are
initialized from scratch. See https://github.com/richzhang/PerceptualSimilarity
and https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html.
