# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog

generator = Llama.build(
        ckpt_dir="llama-2-70b-chat",
        tokenizer_path="tokenizer.model",
        max_seq_len=512,
        max_batch_size=8,
    )

def get_response( dialogs: str):   
    
    results =generator.chat_completion(
        dialogs,
        max_gen_len=None, # The maximum length of generated sequences. If None, it will be set to the model's max sequence length. Defaults to None.
        temperature=0.6,    # The temperature value for controlling randomness in generation.
        top_p=0.9,  # The top-p sampling parameter for controlling diversity in generation. Defaults to 0.9.
    )

    return translate_text(f"{results[0]['generation']['content']}", "arabic")


def create_dialog(crisis, sector,is_injuries=0):
    #translate function to english
    #crisis=translate_text(crisis, "english")
    #sector=translate_text(sector, "english")
    # create prompt
    # Construct the query
    if is_injuries:
        prompt = f"Please list the suggested actions from a {sector} perspective that a nation should take in response to a {crisis} Taking into account the presence of dead and injured  people"
    else:
        prompt = f"Please list the suggested actions from a {sector} perspective that a nation should take in response to a {crisis}"
    
    # create system guidance
    system_guidance= f"act as an Egyptian political man without emojis."

    if sector == "health":
        system_guidance =  f"It is imperative to consistently make decisions that safeguard the health of citizens, mitigate the risk of potential natural disasters, and offer recommendations aimed at preventing such occurrences. So, give me a short answer about {crisis} without emojis."
    elif sector == "national security":
        system_guidance =  f"Your decisions should be geared towards ensuring Egyptian national security and the safety of the people, all while respecting international and diplomatic boundaries. So, give me a short answer about {crisis} without emojis."
    elif sector == "economic":
        system_guidance =  f"You are required to make decisions that serve the best interests of the nation's economy, considering prevailing economic conditions, the stock market, and the guidance of the Central Bank of Egypt. Give a short answer about {crisis} without emojis."
    elif sector == "education":
        system_guidance =f" You are tasked with making decisions that prioritize the best interests of the students, fostering their learning experiences, and offering recommendations that contribute to the advancement of education. So, give me a short answer about {crisis} without emojis."
    elif sector == "foreign policy":
        system_guidance = f" The decision must align accurately with the policies and laws of the Arab Republic of Egypt, ensuring it is in the best interest of the country, all the while upholding strong international diplomatic relations. So, give me a short answer about {crisis} without emojis."
    elif sector == "media":
        system_guidance = f" Compliance with the regulations set forth by the unions associated with this sector is imperative, encompassing both audio-visual and written domains. So, give me a short answer about {crisis} without emojis."


    # create dialog
    
    dialogs: List[Dialog] = [
        [
            #{"role": "system", "content": system_guidance },  
            {"role": "user", "content": prompt}
        ]
    ]

    return dialogs


#if __name__ == "__main__":
#    fire.Fire(main)
print(get_response(create_dialog("The rise in the price of the dollar", "economic",is_injuries=0)))
