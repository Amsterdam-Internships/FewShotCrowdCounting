# Few-shot learning for fast scene adaptation of crowd counting models.
Suppose we wish to count the number of people in an image taken with a surveillance camera. The usual approach would be to place the camera, annotate a large number of images, and then train a model on these images. One issue here is that annotating images (in this case, placing dots on the heads of people) is a time-consuming endeavour. This project is about alleviating this issue a bit by using few-shot learning techniques. Few-shot learning in the context of crowd counting is to adapt a crowd counting model to a novel camera scene using minimal (1 to 5, or sometimes even 0) annotated training examples. As such, we as the municipality would no longer be required to annotate a large number of images whenever we place a new camera.


More information will follow once available.



If applicable, you can also show an example of the final output.

![](media/examples/CrowdCountingExample.jpeg)

---


## Project Folder Structure

Explain a little bit what's where so people can find their way around. For exampole:

There are the following folders in the structure:

1) [`resources`](./resources): Random nice resources, e.g. [`useful links`](./resources/links.md)
1) [`src`](./src): Folder for all source files specific to this project
1) [`media`](./media): Folder containing media files (icons, video)
1) ...

---


## Installation

Explain how to set up everything. 
Let people know if there are weird dependencies - if so feel free to add links to guides and tutorials.

A person should be able to clone this repo, follow your instructions blindly, and still end up with something *fully working*!

1) Clone this repository:
    ```bash
    git clone https://github.com/Amsterdam-Internships/InternshipAmsterdamGeneral
    ```

2) Install all dependencies:
    ```bash
    pip install -r requirements.txt
    ```
---


## Usage

Explain example usage, possible arguments, etc. E.g.:

To train... 


```
$ python train.py --some-importang-argument
```

---


## How it works

Explain roughly how the code works, what the main components are, how certain crucial steps are performed...

---
## Acknowledgements


Don't forget to acknowledge any work by others that you have used for your project. Add links and check whether the authors have explicitly stated citation preference for using the DOI or citing a paper or so. 
For example:

Our code uses [YOLOv5](https://github.com/ultralytics/yolov5) [![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)

