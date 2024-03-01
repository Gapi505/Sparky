import yaml
from dotenv import load_dotenv
import os
import discord
import time
import asyncio
load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')
INACTIVITY_PERIOD = 60 * 2
CHECK_PERIOD = 60

from llama_cpp import Llama
llm = Llama.from_pretrained(
    repo_id="TheBloke/llama2_7b_chat_uncensored-GGUF",
    filename="*Q4_K_M.gguf",
    verbose=False,
    n_gpu_layers=-1, # Uncomment to use GPU acceleration
)

prompt_template = """
### {user}
{user_prompt}

### SPARKY:
"""
history_template = """
### {user}
{message}
"""

# Inference can also be done using transformers' pipeline

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

        message_strip_len = 8 if command.startswith('!sparky') else 3
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
            history += history_template.format(user="HUMAN "+str(m.author).upper(), message=m.content)
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
        response = llm(
            full_prompt, # Prompt
            max_tokens=486, # Generate up to 32 tokens, set to None to generate up to the end of the context window
            stop=["### HUMAN", "\n"], # Stop generating just before the model would generate a new question
            echo=True # Echo the prompt back in the output
        ) # Generate a completion, can also call create_completion
        print(response)
        response = response["choices"][0]["text"]
        print(response)
        filtered_response = response[len(full_prompt):]
        end_time = time.time()
        print("response time:", end_time - start_time)

        await message.channel.send(filtered_response)
client.run(BOT_TOKEN)