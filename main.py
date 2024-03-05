import torch
from diffusers import StableDiffusionPipeline
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

async def unload_llm():
    global llm
    llm = None	
async def load_llm():
    global llm
    torch.cuda.empty_cache()
    llm = Llama.from_pretrained(
    repo_id="TheBloke/dolphin-2.1-mistral-7B-GGUF",
    filename="*Q4_K_M.gguf",
    verbose=True,
    n_gpu_layers=-1,
    n_ctx=4096,
)

async def LLM_Response(message, mode):
    if not lock.locked():
        async with lock:
            global llm
            # return if another LLM thread is running

            user_prompt = message.content

            start_time = time.time()

            history_limits = {
                Mode.DEFAULT: 10,
                Mode.CODE: 20,
                Mode.IMAGEN_PROMPTING: 0
            }

            message_history = []
            async for m in message.channel.history(limit=history_limits.get(mode, 10)):
                message_history.insert(0, m)
            
            #discard messages if they are after the command !split
            for i, m in enumerate(message_history):
                if m.content.startswith("!split"):
                    message_history = message_history[i-1:]
                    break

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
            else:
                return

            
            author = message.author
            full_prompt = prompt_template.format(system_prompt=system_prompt, history=history, user=author, user_message=user_prompt)

            start_time = time.time()
            print(full_prompt)
            response = llm(
                full_prompt, # Prompt
                max_tokens=484, # Generate up to 32 tokens, set to None to generate up to the end of the context window
                stop=["<|im_start|>user"], # Stop generating just before the model would generate a new question
                echo=True,
                temperature=0.80, # Echo the prompt back in the output
            ) # Generate a completion, can also call create_completion
            print(json.dumps(response, indent=1))
            response = response["choices"][0]["text"]
            filtered_response = response[len(full_prompt):].rstrip("[/RESP]")
            end_time = time.time()
            print("response time:", end_time - start_time)

            await message.channel.send(filtered_response)
            return filtered_response
    else:
        print("Another instance of LLM_Response is being processed.")
        message.channel.send("Wait you impatient fuck")
        return

async def imagen(message,prompt):
    if not lock.locked():
        async with lock:
            global imagen_pipe
            await unload_llm()
            imagen_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16).to("cuda")
            image = imagen_pipe(prompt).images[0]
            image.save(f'image.png')
            await message.channel.send(file=discord.File('image.png'))
            imagen_pipe = None
            await load_llm()
    else:
        print("Another instance of imagen is being processed.")
        message.channel.send("Wait you impatient fuck")
        return


@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')
    await client.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="!help"))

async def keep_typing(channel):
    while True:
        await channel.typing()
        print("typing")
        await asyncio.sleep(5)

def validate_command(command_prefix):
    valid_singlecharcommands = ["!","s", "c"]
    valid_longsubcommands = ["img"]
    valid_commands = [ "help", "imagen", "split"]
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
    if message.author == client.user:
        return

    mode = Mode.DEFAULT

    command = message.content.lower()
    if command.startswith('!'):
        # gets how many characters the command is untill the first space. Handle single word commands by setting prefix_len to the length of the command
        words = command.split()
        command_prefix = words[0] if words else ''                    
        
        if not validate_command(command_prefix):
            await message.channel.send("Invalid command")
            return

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
            typing_task = asyncio.create_task(keep_typing(message.channel))
            typing_task
            print("imagen")
            await imagen(message, message.content[7:]) 
            typing_task.cancel()
            return
        

        if 's' in command_prefix:
            mode = Mode.DEFAULT
            if 'c' in command_prefix:
                mode = Mode.CODE
            if 'img' in command_prefix:
                mode = Mode.IMAGEN_PROMPTING
                typing_task = asyncio.create_task(keep_typing(message.channel))
                typing_task

                resp = await LLM_Response(message, mode)
                await imagen(message, resp)
                typing_task.cancel()
                return
        else:
            return

        
        
        typing_task = asyncio.create_task(keep_typing(message.channel))
        typing_task

        await LLM_Response(message, mode)
        
        typing_task.cancel()
client.run(BOT_TOKEN)