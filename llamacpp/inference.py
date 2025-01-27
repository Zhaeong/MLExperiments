# https://sausheong.com/writing-fan-fiction-with-llms-4a76dc6744ba?sk=441d6289d471925a54b59d8356fd31a7

from llama_cpp import Llama

llm = Llama(
      model_path="C:\\Users\\Owen\\.lmstudio\\models\\QuantFactory\\gemma-2-Ifable-9B-GGUF\\gemma-2-Ifable-9B.Q4_0.gguf",
      n_gpu_layers=20, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      n_ctx=8192, # Uncomment to increase the context window default is 512
)

prompt_outline = '''
Given the outline of the plot below, elaborate on the plot and provide the
characters in the story and locations of where the story plays out. Give
names to major characters and locations. Provide step by step progression
of the story. With an emphasis on titillation.

Outline:
--
a big story
'''
plot = ""
with open('plot.txt', 'r') as file:
    # Read the content of the file
    plot = file.read()

story = ""
with open('story.txt', 'r') as file:
    # Read the content of the file
    plot = file.read()


prompt_begin = f'''
[overall plot]
{plot}
---
Write the first section of the story in detail, given the overall plot, 
setting the stage for the rest of the story. Keep the story open-ended
such that it can be easily continued in the next section. When possible, 
provide the motivations of the various characters in the story.

Start with "1. *Crash & Captivation:".
'''

prompt_middle = f'''
[overall plot]
{plot}
---
[story so far]
{story}
---
Continue fleshing the story and create the next section, following the 
overall plot and the story till now. Keep the section open-ended such that 
it can be easily continued in the next section. When possible, provide
the motivations of the various characters in the story.

Start with "4. *Havenwood's Hush & The Zenith's Glow:*".
'''

prompt_end = f'''
[overall plot]
{plot}
---
[story so far]
{story}
---
End the story with a twist and create the last section following the overall
plot and the story till now. When possible, provide the motivations of the 
various characters in the story. Wrap up the entire story as this is the 
last section. 

Start with "5. *Unraveling & Revelation:*".
'''


output = llm(
      prompt_end, # Prompt
      max_tokens=None, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["Q:"], # Stop generating just before the model would generate a new question
      echo=False # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
print(output)

output_string = output['choices'][0]['text']
print(output_string)



with open('output.txt', 'w') as file:
    # Write the string to the file
    file.write(output_string)

