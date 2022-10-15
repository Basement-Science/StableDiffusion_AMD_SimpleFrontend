from dml_onnx_SF import *
import pickle
import random

# load or create file that holds some persistent variables
saveFileName = "saveState.pickle"
imageNumber = 1
runCount = 1
logFileName = "output.log"
try:
    saveFile = open(saveFileName, "rb")
    imageNumber = pickle.load(saveFile)
    try: 
        runCount = pickle.load(saveFile)
    except (EOFError) as e: # file does not contain 2nd variable
        #print(f"exception: {e}")
        saveFile = open(saveFileName, "wb")
        pickle.dump(runCount, saveFile)
except (OSError, IOError, EOFError) as e: # file does not exist / is not accessible
    #print(f"exception: {e}")
    saveFile = open(saveFileName, "wb")
    pickle.dump(imageNumber, saveFile)
#print(f"imageNumber: {imageNumber}, runCount: {runCount}")

try: 
    logFile = open(logFileName, "a")
except (OSError, IOError) as e:
    print("unable to write to log file")

# ########################################################################################## #
class simpleFrontend:
    def createImage(self, prompt, seed, inferenceSteps = 50, Gscale = 7.5):
        global imageNumber
        global runCount
        #if self.lastSeed != seed or self.lastGscale != Gscale: # causes problems sometimes for whatever reason.
        torch.manual_seed(seed)
            
        outputFilePath = f"outputs/img_{imageNumber}_seed{seed}_steps{inferenceSteps}_Gscale{Gscale}_run{runCount}.png"
        print(f"File will be: {outputFilePath}")
        
        image = self.pipe(prompt, height=512, width=512, num_inference_steps=inferenceSteps, guidance_scale=Gscale, eta=0.0, execution_provider="DmlExecutionProvider")["sample"][0]
        image.save(outputFilePath)
        print("image saved successfully.")
        
        logFile.write(f"{outputFilePath} \t{prompt}\n")
        logFile.flush()
        
        imageNumber += 1
        self.lastSeed = seed
        self.lastGscale = Gscale
    # end createImage
    
    def __init__(self):
        self.lastSeed = -1
        self.lastGscale = -1
        def dummy_checker(images, **kwargs): return images, False
        self.lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
        self.pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=self.lms, use_auth_token=True)
        # self.pipe.safety_checker = dummy_checker # disable nsfw filter - not sure if this works.
    # end __init__

# ########################################################################################## #
# Set up task(s) in these functions below:
# return an array of values you want to use for each parameter. Uncomment one return line per function.
# NOTE: you can abort the script using ctrl+c in the console window at any time.
def getPrompts():
    return [
    "a photo of an astronaut riding a horse on mars"]

def getSeeds():
    #return [0] # single value
    #return [464, 488, 764123, 55] # multiple specific values
    #return [i + 42 for i in range(10)] # range of values, starting at an offset
    return [random.randint(0, 999999) for i in range(2)] # a bunch of random value
    #return [78, 79, 80, 83, 86] + [i + 65 for i in range(4)] # combination of above
    
def getInferenceSteps():
    return [40] # single value
    #return [35, 45, 50, 60] # multiple specific values
    #return [15 + (i*5) for i in range(10)] # range of values

def getGscale():
    return [6.5] # single value
    #return [4 + (i*2) for i in range(6)]
    #return [5.0 + (i*0.5) for i in range(4)]

# ########################################################################################## #
# run the combinations of parameters. No caching is done, so each combination will take its time.
try:
    FE = simpleFrontend()
    
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
# Image width and height is set to 512x512
# If you need images of other sizes (size must be divisible by 8), make sure to save the model with that size using save_onnx.py (last lines)
# For example, if you need width=768 and height=512, change save_onnx.py with width=768 and height=512 and also change it in this file at line 41
# Default values are height=512, width=512, num_inference_steps=50, guidance_scale=7.5, eta=0.0, execution_provider="DmlExecutionProvider"
