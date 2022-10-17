from dml_onnx_SF import *
from save_onnx_SF import convert_to_onnx
import pickle
import random
import hashlib
import os
from typing import Optional

# ########################################################################################## #
# More compatible models can potentially be added here in future
models = { # dictionary of short display names for diffusion models
        "sd14": "CompVis/stable-diffusion-v1-4",
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
        print("generating text parsing model...")
        torch.manual_seed(42)
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
# Set up task(s) in these functions below:
# return an array of values you want to use for each parameter. Uncomment one return line per function.
# NOTE: you can abort the script using ctrl+c in the console window at any time.
def getPrompts():
    tokens = ["blue"]
    return [f"a huge {token} and white sci-fi cyberpunk robot raptor with 2 legs, walking through a big lake, water splashing and waves, super detailed and intricate, golden ratio, sharp focus" for token in tokens]

def getSeeds():
    return [459] # single value
    #return [464, 488, 764123, 55] # multiple specific values
    #return [i + 44 for i in range(6)] # range of values, starting at an offset
    #return [random.randint(0, 999999) for i in range(5)] # a bunch of random value
    #return [78, 79, 80, 83, 86] + [i + 65 for i in range(4)] # combination of above
    
def getInferenceSteps():
    return [30] # single value
    #return [25, 35, 50, 75] # multiple specific values
    #return [15 + (i*5) for i in range(10)] # range of values

def getGscale():
    return [8.5] # single value
    #return [5 + (i*2) for i in range(3)]
    #return [5.0 + (i*0.5) for i in range(10)]

# ########################################################################################## #
# run the combinations of parameters. No caching is done, so each combination will take its time.
try:
    # Specify the model and parameters. Changing these will require recompilation of the model.
    model_settings = ModelSettings(width = 512, height = 512, model = "sd14")
    #model_settings = ModelSettings(width = 768, height = 512, model = "sd14")
    #model_settings = ModelSettings(width = 768, height = 512, model = "wd")
    
    FE = simpleFrontend(model_settings)
    
    for prompt in getPrompts():
        print(f"prompt: {prompt}")
        for seed in getSeeds():
            for inferenceSteps in getInferenceSteps():
                for G_scale in getGscale():
                    FE.createImage(prompt, seed, inferenceSteps, G_scale)
finally: # at program exit, save persistent variables to 'saveFile'
    logFile.close()
    with open(saveFileName, "wb") as saveFile:
        pickle.dump(imageNumber, saveFile)
        pickle.dump(runCount + 1, saveFile)

# Works on AMD Windows platform
# Default values are height=512, width=512, num_inference_steps=50, guidance_scale=7.5, eta=0.0, execution_provider="DmlExecutionProvider"
