# StableDiffusion SimpleFrontend for AMD/Windows - v2.2
A script-based "Frontend" for Stable Diffusion on AMD GPUs, to extend [harishanand95's](https://github.com/harishanand95/diffusers/blob/dml/examples/inference/dml_onnx.py) example script.

Supports text2image only. Untested on operating systems other than Windows.

## What can it do?
- Automatically **names all output images** with a unique number, parameters used, and the run number. Does not reassign any numbers after the respective image has been moved/renamed/deleted.
- Logs all ***prompts*** used for each image file additionally in a text file `output.log`. This way you can recreate your images - or variations of them - later on. - **v2.2: You can now recreate images just based on the image number.**
- Allows varying any prompt or parameter combinations and enqueueing them as a batch to be generated. Includes instructions on how to do so.
- Can be aborted at any time.
- Support for other models (Waifu Diffusion, SD 1.5) and resolutions, with Automatic Model rebuilding.

## How to set up
1. Follow instructions to generate your first image using the [harishanand95](https://github.com/harishanand95/diffusers/tree/dml) model. See for example [this guide](https://gitgudblog.vercel.app/posts/stable-diffusion-amd-win10).
2. If you ran into any problems, resolve them now. 
3. Download this repository as zip, and copy the `*.py` files from this repository to your `diffusers\examples\inference` folder. <br>
If you prefer to use `git clone`, you could clone this repo into another folder and symlink the script files over.

## How to use
1. Before trying to use it, always make sure you have activated the python environment in your current terminal. To do so, open a Terminal and `cd` to `diffusers\amd_venv\Scripts` and run `activate`. Then `cd` to `diffusers\examples\inference`.

2. **Open `simpleFrontend.py` in a text editor** _(preferrably in one with python syntax highlighting, like Notepad++ or Visual Studio Code)_ and add prompts to the `worksets` list (near the end of the file, after `# define the work to be done here.`). 
   - You can see examples _(commented lines starting with `#`)_ of how to create a `PromptPermutationSet` object in the file. Each one represents a single prompt OR any permutation of parameters. _For example one prompt text and 6 different seeds to create 6 images._ 
   - You can add as many `PromptPermutationSet` objects as you want, if you want to run different prompts with different parameters without restarting the script. They will be processed in order.
<br><br>
3. Each parameter (including the prompt text) of each `PromptPermutationSet()` can be either a single value OR a list of values. 
4. The different parameters are as follows:
    - `prompts` -> The text prompt(s) that describe what you want in english words.  
    - `seeds` -> A number that causes vast variations of the image.
    - `inferenceSteps` -> Number of steps taken to '_refine_' the image. Has a big impact on processing time, but diminishing impact on the image after a certain value.
    - `getGscale()` -> This number changes details - or adds or removes them - in an image. Important for finalizing an image to your taste.
<br><br>
1. **v2.2:** You can now also recreate previous image prompts - and change some parameters - using the `PromptPermutationSet_fromPreviousImgNumber()` function. This regenerates a ``PromptPermutationSet`` object from the information in your `output.log` file and therefore works even if you deleted or changed or moved that image. ___(also compatible with log files from v2.0)___
2. Further down in the file, the following line specifies which **model** to use and at which **resolution**. You can change this like in the commented lines below it.
   ```python
   model_settings = ModelSettings(width = 512, height = 512, model = "sd14")
   ```
3. Once everything is set up, save the file and simply run `simpleFrontend.py` in the Terminal with the activated python environment. The script will show some progress info. <p>
You dont have to run the script until completion - if you want to change anything, abort the script by pressing `ctrl+c` in the Terminal, make changes in the script and run it again _(dont forget to save)_. You can also edit the script while it is still running - The changes take effect the next time you start it.</p>
5. Your generated images are stored in the `outputs` folder in `diffusers\examples\inference`. 
<br>The filename tells you the parameters except the promt text. The prompt can be found in `output.log` outside the `outputs` folder. The `run` number tells you whether images were generated in the same batch or not.

## Notes
- The script will store some variables in the file `saveState.pickle`. Without this, it will loose track of image number and run number and could overwrite existing image files. Additionally, `modelSettings.pickle` is used to store information about the currently compiled model. Without it, the model will be recompiled on next run.
- By default, image resolution is 512x512. On AMD cards most higher resolutions cannot be generated at the moment or have issues. However 768x512 should work, provided you have enough VRAM. 
- `dml_onnx_SF.py` and `save_onnx_SF.py` contain almost no changes over [harishanand95's example script](https://github.com/harishanand95/diffusers/blob/dml/examples/inference/dml_onnx.py). They only omit the generation of an example image at the end, or the actual generation of the model data in the latter file. You can have these files and their original (non-`_SF` versions) side by side this way. If you want, you can make these modification yourself in the original and then change the first lines of `simpleFrontend.py` to import from those files instead.

## Changelog
### v1.0 first release
### v2.0 Major update
- added Support for other models (e.g. **Waifu Diffusion**).
- added Automatic Model rebuilding when relevant parameters were changed.
- added log file `output.log` that stores information about how to regenerate previous images.
### v2.2 Batch processing improvements
- changed workflow
- batch jobs of higher complexity can now be created using independent `PromptPermutationSet` objects.
- added ability to re-create previously generated images much more easily using `PromptPermutationSet_fromPreviousImgNumber()`.
- added option for **Stable Diffusion 1.5** model