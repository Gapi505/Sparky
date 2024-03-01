import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import torch
import os
import discord
import time
import asyncio
load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')
INACTIVITY_PERIOD = 60 * 2
CHECK_PERIOD = 60

model_name_or_path = "TheBloke/llama2_7b_chat_uncensored-GPTQ"
# To use a different branch, change revision
# For example: revision="main"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=True,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

prompt_template = """
###{user}
{user_prompt}

###SPARKY:
"""
history_template = """
###{user}
{message}
"""

# Inference can also be done using transformers' pipeline

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=484,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

#print(pipe(prompt_template)[0]['generated_text'])

# set up the discord bot
intent = discord.Intents.default()
intent.message_content = True
intent.typing = True

client = discord.Client(intents=intent)


@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')
    await client.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="!sparky or !s"))

async def keep_typing(channel):
    while True:
        await channel.typing()
        print("typing")
        await asyncio.sleep(5)

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    command = message.content.lower()
    if command.startswith('!sparky') or command.startswith('!s'):

        message_strip_len = 7 if command.startswith('!sparky') else 2
        user_prompt = message.content[message_strip_len:]

        start_time = time.time()

        message_history = []
        async for m in message.channel.history(limit=5):
            message_history.append(m)
        history = ""
        for m in message_history:
            if m.author == client.user:
                history += history_template.format(user="SPARKY", message=m.content)
                continue
            history += history_template.format(user=str(m.author).upper(), message=m.content)
        end_time = time.time()
        print("history time:", end_time - start_time)
            
            
        print("loading personality")
        with open("personality.yaml", "r") as file:
            personality_configuration = yaml.safe_load(file)
        print("loaded personality")

        author = str(message.author)
        if author in personality_configuration:
            print("Using custom personality for", author)
            full_prompt = personality_configuration[author]
        else:
            print("Using default personality for", author)
            full_prompt = personality_configuration["default"]
        full_prompt += history
        full_prompt += prompt_template.format(user=author.upper(), user_prompt=user_prompt)

        start_time = time.time()
        response = pipe(full_prompt)[0]['generated_text']
        filtered_response = response[len(full_prompt):]
        end_time = time.time()
        print("response time:", end_time - start_time)

        await message.channel.send(filtered_response)
client.run(BOT_TOKEN)