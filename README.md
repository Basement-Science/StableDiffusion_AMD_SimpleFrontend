# StableDiffusion SimpleFrontend for AMD/Windows
A script-based "Frontend" for Stable Diffusion on AMD GPUs, to extend [harishanand95's](https://github.com/harishanand95/diffusers/blob/dml/examples/inference/dml_onnx.py) example script.

Supports text2image only. Untested on operating systems other than Windows.

## What can it do?
- Automatically **names all output images** with a unique number, parameters used, and the run number. Does not reassign any numbers after the respective image has been moved/renamed/deleted.
- Logs all ***prompts*** used for each image file additionally in a text file `output.log`. This way you can recreate your images - or variations of them - later on.
- Allows varying any prompt or parameter combinations and enqueueing them as a batch to be generated. Includes instructions on how to do so.
- Can be aborted at any time.

## How to set up
1. Follow instructions to generate your first image using the [harishanand95](https://github.com/harishanand95/diffusers.git) model. See for example [this guide](https://gitgudblog.vercel.app/posts/stable-diffusion-amd-win10).
2. If you ran into any problems, resolve them now. 
3. Download this repository as zip, and copy `simpleFrontend.py` and `dml_onnx_SF.py` from this repository to your `diffusers\examples\inference` folder. <br>
If you prefer to use `git clone`, you could clone this repo into another folder and symlink the script files over.

## How to use
1. Before trying to use it, always make sure you have activated the python environment in your current terminal. To do so, open a Terminal and `cd` to `diffusers\amd_venv\Scripts` and run `activate`. Then `cd` to `diffusers\examples\inference`.

2. **Open `simpleFrontend.py` in a text editor** _(preferrably in one with python syntax highlighting, like Notepad++ or Visual Studio Code)_ and modify the `return` values of the functions listed below. One image will be generated for each combination of `prompt`, `seed`, `inferenceSteps` and `guidance_scale (Gscale)` values. See commented lines _(starting with `#`)_ for examples. 
<br>**The following functions are intended to be changed:**
    - `getPrompts()` -> The text prompt(s) that describe what you want.  
    - `getSeeds()` -> A number that generates vast variations of the image.
    - `getInferenceSteps()` -> Number of steps taken to '_refine_' the image. 
    - `getGscale()` -> This number changes details - or adds or removes them - in an image. Important for finalizing an image to your taste.
<br><br>
3. Once everything is set up, save the file and simply run `simpleFrontend.py` in the Terminal with the activated python environment. The script will show some progress info. <p>
You dont have to run the script until completion - if you want to change anything, abort the script by pressing `ctrl+c` in the Terminal, make changes in the script and run it again _(dont forget to save)_. You can also edit the script while it is still running - The changes take effect the next time you start it.</p>
4. Your generated images are stored in the `outputs` folder in `diffusers\examples\inference`. 
<br>The filename tells you the parameters except the promt text. The prompt can be found in `output.log` outside the `outputs` folder. The `run` number tells you whether images were generated in the same batch or not.

## Notes
- The script will store some variables in the file `saveState.pickle`. Without this, it will loose track of image number and run number and could overwrite existing image files.
- By default, image resolution is 512x512. On AMD cards most higher resolutions cannot be generated at the moment or have issues. But if you want to try, follow the instructions at the very end of `simpleFrontend.py`. 
On my RX 6800 XT I am able to use 768x512 resolution, but no higher.
- `dml_onnx_SF.py` contains almost no changes over [harishanand95's example script](https://github.com/harishanand95/diffusers/blob/dml/examples/inference/dml_onnx.py). It only omits the generation of an example image at the end. You can have `dml_onnx_SF.py` and the original `dml_onnx.py` side by side this way. If you want, you can make this modification yourself in the original and then change the first line of `simpleFrontend.py` to import from that file instead.