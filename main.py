import torch
import datetime
import json
import yaml
from dotenv import load_dotenv
import os
import discord
import time
import asyncio
from enum import Enum
import re
from diffusers import AutoPipelineForText2Image
import torch


load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')
INACTIVITY_PERIOD = 60 * 2
CHECK_PERIOD = 60

from llama_cpp import Llama
llm = Llama.from_pretrained(
    repo_id="TheBloke/dolphin-2.1-mistral-7B-GGUF",
    filename="*Q4_K_M.gguf",
    verbose=True,
    n_gpu_layers=-1,
    n_ctx=4096,
)
imagen_pipe = None


prompt_template = """
<|im_start|>
{system_prompt}<|im_end|>

{history}
<|im_start|>user {user}
{user_message}<|im_end|>
<|im_start|>assistant
"""
history_template = """
<|im_start|>user {user}
{message} <|im_end|>
<|im_start|>assistant
{asistant_message} <|im_end|>"""

# Inference can also be done using transformers' pipeline

#print(pipe(prompt_template)[0]['generated_text'])

# set up the discord bot
intent = discord.Intents.default()
intent.message_content = True
intent.typing = True

client = discord.Client(intents=intent)

lock = asyncio.Lock()

class Mode(Enum):
    DEFAULT = 0
    CODE = 1
    IMAGEN_PROMPTING = 2
    CONSPIRACY = 3

dev_options = None
with open("dev_options.yaml", "r+", encoding="utf8") as file:
    dev_options = yaml.safe_load(file)

model_loaded = True
async def unload_llm():
    global llm
    global model_loaded
    model_loaded = False
    del llm
    torch.cuda.empty_cache()
async def load_llm():
    global llm
    global model_loaded
    model_loaded = True
    torch.cuda.empty_cache()
    llm = Llama.from_pretrained(
    repo_id="TheBloke/dolphin-2.1-mistral-7B-GGUF",
    filename="*Q4_K_M.gguf",
    verbose=True,
    n_gpu_layers=-1,
    n_ctx=8096,
)
async def reload_llm():
    await unload_llm()
    await load_llm()

async def LLM_Response(message, mode):
    if lock.locked():
        print("Another instance of LLM_Response is being processed.")
        await message.channel.send("Wait you impatient fuck")
        return
    
    async with lock:
        async with message.channel.typing():
            global llm
            # return if another LLM thread is running

            user_prompt = message.content

            start_time = time.time()

            history_limits = {
                Mode.DEFAULT: 20,
                Mode.CODE: 30,
                Mode.IMAGEN_PROMPTING: 0,
                Mode.CONSPIRACY: 20
            }

            message_history = [msg async for msg in message.channel.history(limit=history_limits[mode])]
            for i, m in enumerate(message_history):
                print(m.author, m.content)
                if m.content.startswith("!split"):
                    message_history = message_history[:i+1]
                    break
            message_history.reverse()
            print(message_history)

            history = ""
            for i, m in enumerate(message_history):
                if i >= len(message_history) - 1:
                    break
                print(m.author, m.content)
                if m.author == client.user:
                    continue
                if message_history[i+1].author == client.user:
                    history += history_template.format(user=m.author, message=m.content, asistant_message=message_history[i+1].content)
                else:
                    history += history_template.format(user=m.author, message=m.content, asistant_message="(silence)")
            end_time = time.time()
            print("history time:", end_time - start_time)
                
                
            print("loading sys prompt")
            with open("system_prompt.yaml", "r", encoding="utf8") as file:
                system_prompts = yaml.safe_load(file)
            print("loaded sys prompt")
            if mode == Mode.DEFAULT:
                system_prompt = system_prompts["default"]
            elif mode == Mode.CODE:
                system_prompt = system_prompts["code"]
            elif mode == Mode.IMAGEN_PROMPTING:
                system_prompt = system_prompts["imagen"]
            elif mode == Mode.CONSPIRACY:
                system_prompt = system_prompts["conspiracy"]
            else:
                return

            
            author = message.author
            full_prompt = prompt_template.format(system_prompt=system_prompt, history=history, user=author, user_message=user_prompt)
            start_time = time.time()
            print(full_prompt)
            response = llm(
                full_prompt, # Prompt
                max_tokens=460, # Generate up to 32 tokens, set to None to generate up to the end of the context window
                stop=["<|im_start|>user"], # Stop generating just before the model would generate a new question
                echo=True,
                temperature=dev_options["temp"], # Echo the prompt back in the output
            ) # Generate a completion, can also call create_completion
            print(json.dumps(response, indent=1))
            response = response["choices"][0]["text"]
            filtered_response = response[len(full_prompt):].rstrip("[/RESP]")
            end_time = time.time()
            print("response time:", end_time - start_time)

            await message.channel.send(filtered_response)
            return filtered_response

async def imagen(message,prompt):
    if not lock.locked():
        async with lock:
            async with message.channel.typing():
                global imagen_pipe
                await unload_llm()
                imagen_pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
                #imagen_pipe = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", filename="*emaonly.safetensors",torch_dtype=torch.float16, variant="fp16")
                imagen_pipe.to("cuda")

                imagen_pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0].save("image.png")
                print("image generated")
                await message.channel.send(file=discord.File("image.png"))
                del imagen_pipe
                torch.cuda.empty_cache()
                await load_llm()
    else:
        print("Another instance of imagen is being processed.")
        await message.channel.send("Wait you impatient fuck")
        return

model_last_used = time.time()

async def model_timeout(timeout):
    global model_last_used
    global model_loaded
    while True:
        if time.time() - model_last_used > timeout and model_loaded == True:
            print("Model timeout.")
            await unload_llm()
            # Add code here to handle the timeout, e.g. unload the model
        await asyncio.sleep(30)  # Sleep for a while before checking again

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')
    asyncio.create_task(model_timeout(180))
    await client.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="!help"))

async def keep_typing(channel):
    while True:
        await channel.typing()
        print("typing")
        await asyncio.sleep(5)


class Type(Enum):
    BOOL = 0
    NUM = 1
    STR = 2

def get_value_type(value):
    if value.lower() == "true" or value.lower() == "false":
        return Type.BOOL
    try:
        float(value)
        return Type.NUM
    except ValueError:
        return Type.STR

# Start the timeout task in the background


def validate_command(command_prefix):
    valid_singlecharcommands = ["!","s", "c"]
    valid_longsubcommands = []
    valid_commands = [ "help", "imagen", "split", "img", "dev"]
    Valid = False
    subcommands = re.split('_|-|:|,|;', command_prefix)
    if command_prefix[1:] in valid_commands:
        Valid = True
    else:
        for subcommand in subcommands:
            if subcommand in valid_longsubcommands:
                Valid = True
            else:
                for char in subcommand:
                    if char in valid_singlecharcommands:
                        Valid = True
                    else:
                        Valid = False
                        break
    return Valid


@client.event
async def on_message(message):
    global model_last_used
    if message.author == client.user:
        return
    
    mode = Mode.DEFAULT

    command = message.content.lower()
    if command.startswith('!'):
        model_last_used = time.time()
        global model_loaded
        if model_loaded == False:
            await load_llm()
        # gets how many characters the command is untill the first space. Handle single word commands by setting prefix_len to the length of the command
        words = command.split()
        command_prefix = words[0] if words else ''               
        
        if not validate_command(command_prefix):
            await message.channel.send("Invalid command")
            return
        if 'dev' in command:
            if str(message.author) != "gapi505":
                print(message.author)
                await message.channel.send("You are not allowed to use this command")
                return
            params = command.split()
            if len(params) < 2:
                await message.channel.send("provide a parameter")
                return
            print(params)
            for param in params[1:]:
                param_name, param_value = param.split(":")
                if param_name in dev_options:
                    value_type = get_value_type(param_value)
                    if value_type == Type.BOOL and type(dev_options[param_name]) == bool:
                        dev_options[param_name] = bool(param_value)
                        await message.channel.send(f"{param_name} set to {dev_options[param_name]}")
                    elif value_type == Type.NUM and (type(dev_options[param_name]) == float or type(dev_options[param_name]) == int):
                        dev_options[param_name] = float(param_value)
                    elif value_type == Type.STR and type(dev_options[param_name]) == str:
                        dev_options[param_name] = param_value
                    
                    await message.channel.send(f"{param_name} set to {dev_options[param_name]}")
                elif param_name == "reload":
                    await reload_llm()
                    await message.channel.send("Model reloaded")
                else:
                    await message.channel.send(f"{param_name} is not a valid parameter")
            with open("dev_options.yaml", "w", encoding="utf8") as file:
                yaml.dump(dev_options, file)

        if 'split' in command:
            return
        if 'help' in command:
            embed = discord.Embed(title="Sparky Instructions", description="A discord bot that uses the dolphin-2.1-mistral-7B-GGUF model to generate text and Stable Diffusion to generate images", color=0x00ff00)
            embed.add_field(name="Activation", value="To activate type **!**\n\n", inline=False)
            embed.add_field(name="Commands", value="**!s** - activates the text bot\n**!help** - displays this help message\n**!split** - makes the bot ignore all mesages sent before this command. Like starting a new chat\n**!imagen** - manual prompt image generation\n\n", inline=False)
            embed.add_field(name="Subcommands / options", value="Subcomands are used after the main command\n Write a seperating character inbetween commands and subcommands \n(**_**, **-**, **:**, **,** or **;**) \nlike this:\n**!s:c [question]**\n\n**c** - optimizes the bot for code\n**img** - image generation with the help of AI for prompting", inline=False)
            await message.channel.send(embed=embed)
            return
        if 'imagen' in command_prefix:
            print("imagen")
            await imagen(message, message.content[7:]) 
            return
        

        if 's' in command_prefix:
            mode = Mode.DEFAULT
            if 'c' in command_prefix:
                mode = Mode.CODE
            if dev_options["consp"] == True:
                mode = Mode.CONSPIRACY
        elif 'img' in command_prefix:
            mode = Mode.IMAGEN_PROMPTING

            resp = await LLM_Response(message, mode)
            #typing_task.cancel()


            #typing_task = asyncio.create_task(keep_typing(message.channel))
            #typing_task
            await imagen(message, resp)
            return
        else:
            return

        await LLM_Response(message, mode)
        
client.run(BOT_TOKEN)