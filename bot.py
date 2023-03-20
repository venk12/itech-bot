import os

import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('MTA4NzI4MjYwMjIyODA3MjQ5OA.GaxehN.iljQfoHNozKgh9eLjrHWOyB5sQ_cwcFXZ5W0vc')

# client = discord.Client(intents=discord.Intents.default())
client = commands.Bot(command_prefix='',intents=discord.Intents.all())

@client.event
async def on_ready():
    print('Connected to bot: {}'.format(client.user.name))
    print('Bot ID: {}'.format(client.user.id))


@client.command()
async def helloworld(ctx):
    await ctx.send('Hello World!')

client.run('MTA4NzI4MjYwMjIyODA3MjQ5OA.GaxehN.iljQfoHNozKgh9eLjrHWOyB5sQ_cwcFXZ5W0vc')