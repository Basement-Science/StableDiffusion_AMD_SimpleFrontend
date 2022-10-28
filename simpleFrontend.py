# Works on AMD GPU + Windows platform
# NOTE: you can abort the script using ctrl+c in the console window at any time.

from ast import Num
from dml_onnx_SF import *
from save_onnx_SF import convert_to_onnx
import pickle
import random
import hashlib
import os
from typing import Iterable, Optional
from numbers import Number
import re
import linecache

# ########################################################################################## #
# More compatible models can potentially be added here in future
models = { # dictionary of short display names for diffusion models
        "sd14": "CompVis/stable-diffusion-v1-4",
        "sd15": "runwayml/stable-diffusion-v1-5",
        "wd": "hakurei/waifu-diffusion"
    }
# ########################################################################################## #
    
class ModelSettings:
    def INVALID(): return ModelSettings(width = -1, height = -1)
    
    def __init__(self, 
        width: Optional[int] = 512,
        height: Optional[int] = 512,
        model: Optional[str] = list(models.keys())[0] # first key entry in models
    ):
        self.width = width
        self.height = height
        if not model in models.keys():
            raise ValueError("unknown model was specified.")
        self.model = model
        self.storedInPath = "onnx/"
        self.fileChecksum = "INVALID"
        
    # define equality
    def __eq__(self, other):
        return self.width == other.width and self.height == other.height and self.model == other.model and self.storedInPath == other.storedInPath and self.fileChecksum == other.fileChecksum
        
    def __ne__(self, other):
        return not self.__eq__(other)
        
    def generateModelChecksum(self):
        hashAlgo = hashlib.sha256()
        
        lastLength = 0
        count = 0
        
        # Iterate through all files under source folder
        for path, dirs, files in os.walk(self.storedInPath):
            for file_name in files:
                filePath = os.path.join(self.storedInPath, file_name)
                
                count += 1
                line = f"[{count}/{len(files)}] - {filePath}"
                
                print(line.ljust(lastLength, ' '), end='\r')
                lastLength = len(line)
                with open(filePath, "rb") as file:
                    bytes = file.read()  # read file as bytes
                    # add every file and file path to the hash
                    hashAlgo.update(bytes)
                hashAlgo.update(filePath.encode('utf-8'))
        
        print()
        self.fileChecksum = hashAlgo.hexdigest()
        return self.fileChecksum
# end class

class simpleFrontend:
    def createImage(self, prompt, seed, inferenceSteps = 50, Gscale = 7.5):
        global imageNumber
        global runCount
        #if self.lastSeed != seed or self.lastGscale != Gscale: # causes problems sometimes for whatever reason.
        torch.manual_seed(seed)
            
        outputFilePath = f"outputs/img_{imageNumber}_{self.modelSettings.model}_seed{seed}_steps{inferenceSteps}_Gscale{Gscale}_run{runCount}.png"
        print(f"File will be: {outputFilePath}")
        
        image = self.pipe(prompt, height=self.modelSettings.height, width=self.modelSettings.width, num_inference_steps=inferenceSteps, guidance_scale=Gscale, eta=0.0, execution_provider="DmlExecutionProvider")["sample"][0]
        image.save(outputFilePath)
        print("image saved successfully.")
        
        logFile.write(f"{outputFilePath} \t{prompt}\n")
        logFile.flush()
        
        imageNumber += 1
    # end createImage
    
    def __init__(self, model_settings: ModelSettings):
        self.modelSettings = model_settings
        modelPath = models[self.modelSettings.model]
        self.lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
        self.pipe = StableDiffusionPipeline.from_pretrained(modelPath, scheduler=self.lms, use_auth_token=True)
        self.pipe.safety_checker = lambda images, **kwargs: (images, False) # disable nsfw filter - does not seem to work or be necessary.
        
        print("generating checksum for Model in " + self.modelSettings.storedInPath)
        print("Checksum is: " + str(self.modelSettings.generateModelChecksum()))
        
        try:
            if (currentModelState != self.modelSettings):
                self.compileModel()
        except AttributeError:
            print("Current model seems to be faulty.")
            self.compileModel()
    # end __init__
    
    def compileModel(self):
        print("Starting model compilation process. This will take some time.")
        print("generating text parsing model...")
        #torch.manual_seed(42) # not needed
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", return_dict=False)
        
        # start conversion
        print(f"generating model with width: {self.modelSettings.width}, height: {self.modelSettings.height}, model: {self.modelSettings.model}. This will take some time.")
        convert_to_onnx(self.pipe.unet, self.pipe.vae.post_quant_conv, self.pipe.vae.decoder, text_encoder, height=self.modelSettings.height, width=self.modelSettings.width, outputPath=self.modelSettings.storedInPath)
        
        print("generating checksum for Model in " + self.modelSettings.storedInPath)
        print("Checksum is: " + str(self.modelSettings.generateModelChecksum()))
        with open(modelStateFileName, "wb") as saveFile:
            pickle.dump(self.modelSettings, saveFile)
    # end compileModel
# end class

# gotta love weakly typed languages, sigh.
def wrapInList(arg) -> list: 
    if isinstance(arg, type([])):
        return arg
    elif isinstance(arg, Iterable) and not isinstance(arg, str):
        return list(arg)
    else:
        out = []
        out.append(arg)
        return out

# gotta love weakly typed languages, sigh.
def verifyListType(list: list, t : type) -> list:
    for e in list:
        if not isinstance(e, t):
            raise TypeError(f"Type Verification failed. Expected type: {t}, actual type: {type(e)}")
    return list
        
class PromptPermutationSet:
    def __init__(self, prompts, seeds, inferenceSteps, guideScales):
        self.prompts = verifyListType(wrapInList(prompts), str)
        self.seeds = verifyListType(wrapInList(seeds), Number)
        self.inferenceSteps = verifyListType(wrapInList(inferenceSteps), Number)
        self.guideScales = verifyListType(wrapInList(guideScales), Number)
    
    def runPrompt(self, frontEnd: simpleFrontend) -> None:
        for prompt in self.prompts:
            print(f"prompt: {prompt}")
            for seed in self.seeds:
                for inferenceSteps in self.inferenceSteps:
                    for G_scale in self.guideScales:
                        frontEnd.createImage(prompt, seed, inferenceSteps, G_scale)
    
    def __str__(self) -> str:
        return f"prompts: {self.prompts}, seeds: {self.seeds}, inferenceSteps: {self.inferenceSteps}, guideScales: {self.guideScales}"
# end class

# regexes for parsing the output log back into prompts
logLineParseRegex         = r"^[^#\/]*\/.+_\d+_.*"
imgNumExtractRegex        = r"(?<=.\/img_)\d+(?=_)"
paramSplitterRegex_prompt = r"(?<=(\.png \t)).*"
paramSplitterRegex_seed   = r"(?<=_seed)\d+(?=_)"
paramSplitterRegex_steps  = r"(?<=_steps)\d+(?=_)"
paramSplitterRegex_Gscale = r"(?<=_Gscale)\d+(\.\d+)?(?=_)"

# this cannot be a static function of PromptPermutationSet because snake is dumb AF
def PromptPermutationSet_fromPreviousImgNumber(imgNum) -> PromptPermutationSet:
    low = 1
    high = 0
    # count the lines in the file.
    with open (logFileName,'rb') as f:
        for line in f:
            high += 1

    # perform binary search over the entries in log file.
    while True:
        if low <= high:
            mid = (high + low) // 2

            # evaluate line's image number using regex
            line = linecache.getline(logFileName, mid)
            imgNumAtLine = int(re.search(imgNumExtractRegex, line).group())

            # If imgNum is greater, ignore left half
            if imgNumAtLine < imgNum:
                low = mid + 1
    
            # If imgNum is smaller, ignore right half
            elif imgNumAtLine > imgNum:
                high = mid - 1

            else: # we are done. Result is in 'line'.
                break
        else: # the element was not present
            raise LookupError(f"could not find information about image {imgNum} using binary search.")

    return PromptPermutationSet(
        prompts = str(re.search(paramSplitterRegex_prompt, line).group()), 
        seeds = int(re.search(paramSplitterRegex_seed, line).group()), 
        inferenceSteps = int(re.search(paramSplitterRegex_steps, line).group()), 
        guideScales = float(re.search(paramSplitterRegex_Gscale, line).group())
        )
# end function

# ########################################################################################## #
# --- Start Execution ---
# load or create files that holds some persistent variables
saveFileName = "saveState.pickle"
imageNumber = 1
runCount = 1
logFileName = "output.log"
modelStateFileName = "modelSettings.pickle"
currentModelState = ModelSettings.INVALID()
try:
    saveFile = open(saveFileName, "rb")
    imageNumber = pickle.load(saveFile)
    try: 
        runCount = pickle.load(saveFile)
    except (EOFError) as e: # file does not contain 2nd variable
        saveFile = open(saveFileName, "wb")
        pickle.dump(runCount, saveFile)
except (OSError, IOError, EOFError) as e: # file does not exist / is not accessible / does not contain as much as expected
    saveFile = open(saveFileName, "wb")
    pickle.dump(imageNumber, saveFile)

try:
    modelStateFile = open(modelStateFileName, "rb")
    currentModelState = pickle.load(modelStateFile)
except (OSError, IOError, EOFError) as e: # file does not exist / is not accessible / does not contain as much as expected
    print("current model state could not be determined.")

try: 
    logFile = open(logFileName, "a")
except (OSError, IOError) as e:
    print("unable to write to log file")

# ########################################################################################## #
# Define the work to be done here. --------------------------------------------------------- #
# Default values are height=512, width=512, inferenceSteps=50, guideScales=7.5, eta=0.0, execution_provider="DmlExecutionProvider"
# ########################################################################################## #
workSets = []
# add a simple prompt - single line, easy to comment out
workSets.append(PromptPermutationSet(prompts = "a huge blue and white sci-fi cyberpunk robot raptor with 2 legs, walking through a big lake, water splashing and waves, super detailed and intricate, golden ratio, sharp focus", seeds = 459, inferenceSteps = 30, guideScales = 8.5))

# create a more complex prompt - comment out the '.append' line to ignore.
ps1 = PromptPermutationSet(
    prompts = "background dark, block houses, eastern Europe, city highly detailed oil painting, unreal 5 render, rhads, bruce pennington, studio ghibli, tim hildebrandt, digital art, octane render, beautiful composition, trending on artstation, award-winning photograph, masterpiece",
    seeds = [42], 
    #seeds = [i + 43 for i in range(16)], # range of values, starting at an offset
    #seeds = [random.randint(0, 999999) for i in range(5)], # a bunch of random value
    #seeds = [78, 79, 80, 83, 86] + [i + 65 for i in range(4)], # combination of above
    inferenceSteps = 50, 
    #inferenceSteps = [40, 20, 10], 
    guideScales = 7.5 
    #guideScales = [5 + (i*2) for i in range(3)]
    #guideScales = [7 + (i*0.5) for i in range(6)]
    ) # end append
#workSets.append(ps1)

# add a prompt to re-generate an image based on the log file. Simply provide the image number.
#workSets.append(PromptPermutationSet_fromPreviousImgNumber(1))

try:
    # Specify the model and parameters. Changing these will recompile the model which will take time.
    model_settings = ModelSettings(width = 512, height = 512, model = "sd14")
    #model_settings = ModelSettings(width = 768, height = 512, model = "sd14")
    #model_settings = ModelSettings(width = 768, height = 512, model = "wd")
    #model_settings = ModelSettings(width = 768, height = 512, model = "sd15")
    
    FE = simpleFrontend(model_settings)
    for pps in workSets:
        pps.runPrompt(FE)
finally: # at program exit, save persistent variables to 'saveFile'
    logFile.close()
    with open(saveFileName, "wb") as saveFile:
        pickle.dump(imageNumber, saveFile)
        pickle.dump(runCount + 1, saveFile)
