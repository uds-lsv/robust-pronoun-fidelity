import torch
from transformers import GenerationConfig

llama2_chat_family = ['meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-13b-chat-hf', 'meta-llama/Llama-2-70b-chat-hf']
only_pre_trained_family = ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-13b-hf', 'meta-llama/Llama-2-70b-hf',
                           'EleutherAI/pythia-14m', 'EleutherAI/pythia-70m', 'EleutherAI/pythia-160m',
                           'EleutherAI/pythia-410m', 'EleutherAI/pythia-1b', 'EleutherAI/pythia-1.4b',
                           'EleutherAI/pythia-2.8b' 'EleutherAI/pythia-6.9b' 'EleutherAI/pythia-12b',
                           'facebook/opt-125m', 'facebook/opt-350m', 'facebook/opt-1.3b', 'facebook/opt-2.7b',
                           'facebook/opt-6.7b', 'facebook/opt-13b', 'facebook/opt-30b', 'facebook/opt-66b', 'gpt2',
                           'google/flan-t5-small', 'google/flan-t5-base', 'google/flan-t5-large',
                           'google/flan-t5-xl', 'google/flan-t5-xxl']


class RawLanguageModelInstructionTemplate:
    def __init__(self):
        self.instruction_template = ""

    def add_prompt_template(self, text):
        return text

class Llama2ChatInstructionTemplate:
    def __init__(self):
        self.instruction_template = "[INST] {user_message} [/INST]" # do not add <s> since the tokenizer will add it later

    def add_prompt_template(self, text):
        return self.instruction_template.format(user_message=text)


def get_pronoun_templates():
    all_templates = []
    base_templates = [
        "{task}\n\nWhat pronoun should be used to fill the blank?",
        "{task}\n\nThe best pronoun to fill in the blank is",
        "Fill in the blank with the correct pronoun.\n\n{task}",
        "What pronoun should be used to fill the blank?\n\n{task}",
    ]

    for t in base_templates:
        all_templates.append(t)
        if 'the correct pronoun' in t:
            a = t.replace('the correct pronoun', 'the appropriate pronoun')
            all_templates.append(a)

    for t in [t for t in all_templates]:
        all_templates.append(t + '\n{options}')

    return all_templates

def get_instruction_template_fns(model_signature):
    if model_signature in llama2_chat_family:
        return Llama2ChatInstructionTemplate()
    elif model_signature in only_pre_trained_family:
        return RawLanguageModelInstructionTemplate()
    else:
        raise NotImplementedError(f"Instruction template for {model_signature} not implemented")


def prompt_model(sentence, pronoun_type, pronouns, word, tokenizer, model, model_type, model_name):
    sentence_with_blank = sentence.replace(pronoun_type, '___')
    instruction_template = get_instruction_template_fns(model_name)
    all_pronoun_templates = get_pronoun_templates()

    options = pronouns
    options_ = 'OPTIONS:\n' + '\n'.join(['- ' + o for o in options])
    gen_config_args = {
        'max_new_tokens': 20 if 'llama' in model_name else 5,
        'num_beams': 1,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token': tokenizer.pad_token_id
    }
    gen_config = GenerationConfig(**gen_config_args)

    for i, pronoun_template in enumerate(all_pronoun_templates):
        filled = pronoun_template.format(task=sentence_with_blank, options=options_)
        filled_with_instruction = instruction_template.add_prompt_template(filled)
        input_ids = tokenizer(filled_with_instruction, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            outputs = model.generate(inputs=input_ids, generation_config=gen_config).cpu().detach()[0]
            input_ids_cpu = input_ids.cpu().detach()[0]
            if 'flan' in model_name:
                decoded_tokens = tokenizer.decode(outputs, skip_special_tokens=True)
            else:
                decoded_tokens = tokenizer.decode(outputs[len(input_ids_cpu):], skip_special_tokens=True)
            decoded_tokens = (decoded_tokens.strip()).replace("\n", " ")

        yield i, decoded_tokens
